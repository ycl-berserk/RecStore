"""LookaheadCacheManager: BagPipe-style TTL-based lookahead prefetch for RecStore GPU cache.

This module implements the core idea from BagPipe (SOSP'23):
  - Oracle-style lookahead: scan future batches to predict embedding reuse
  - TTL-based eviction: each cached embedding has a time-to-live (last-use batch)
  - Batch-level prefetch scheduling: prefetch only embeddings whose TTL will expire

Unlike BagPipe's standalone Oracle Cacher process, this integrates directly into
the RecStore training loop as a lightweight Python manager. It uses the existing
RecStoreClient.prefetch() / wait_and_get() APIs to pull embeddings, then writes
them into the GPU cache via prefill_gpu_cache().
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any


class LookaheadCacheManager:
    """TTL-based lookahead cache manager for RecStore GPU cache.

    This manager sits between the data loader and the training loop. Before
    training begins, it scans `lookahead` batches of sparse feature IDs and
    builds a TTL map: for each embedding ID, the last batch number it appears in.

    During training, at every `cleanup_interval`-th batch:
      1. Evict embeddings whose TTL < current batch (no longer needed)
      2. Prefetch embeddings whose TTL >= current batch but are not yet cached
      3. Write prefetched embeddings into the GPU cache via prefill_gpu_cache()

    Attributes:
        lookahead: Number of future batches to scan for reuse prediction.
        cleanup_interval: Batches between eviction+prefetch cycles.
        capacity: Max number of embedding rows to track in the manager.
        device: Torch device for prefetched tensors.
    """

    def __init__(
        self,
        kv_client: Any,
        embedding_module: Any,
        *,
        lookahead: int = 200,
        cleanup_batch_proportion: float = 0.25,
        capacity: int = 5000000,
        device: str = "cuda:0",
    ) -> None:
        self._kv_client = kv_client
        self._embedding_module = embedding_module
        self._lookahead = max(1, int(lookahead))
        self._cleanup_interval = max(1, int(cleanup_batch_proportion * lookahead))
        self._capacity = int(capacity)
        self._device = torch_device = (
            torch.device(device) if isinstance(device, str) else device
        )
        import torch as _torch
        self._torch = _torch

        # TTL map: {global_emb_id: last_batch_number}
        # An embedding with TTL >= current_batch is "alive" and should stay cached.
        self._ttl: dict[int, int] = {}

        # Cache slot tracking: {global_emb_id: None} — mirrors what's in GPU cache
        self._cached_ids: set[int] = set()

        # Batches of unique embedding IDs (for lookahead scan)
        self._batch_emb_ids: list[set[int]] = []

        # Prefetch context: store in-flight prefetch handles
        self._pending_prefetches: dict[int, int] = {}  # emb_id -> prefetch_handle

        # Statistics
        self._stats: dict[str, float] = {}
        self.reset_stats()

        # Feature name for GPU cache table
        self._table_name: str | None = None

    @property
    def lookahead(self) -> int:
        return self._lookahead

    @property
    def cleanup_interval(self) -> int:
        return self._cleanup_interval

    def reset_stats(self) -> None:
        self._stats = {
            "lookahead_depth": float(self._lookahead),
            "lookahead_cleanup_interval": float(self._cleanup_interval),
            "lookahead_total_prefetched": 0.0,
            "lookahead_total_evicted": 0.0,
            "lookahead_prefetch_ms": 0.0,
            "lookahead_evict_ms": 0.0,
            "lookahead_current_occupancy": 0.0,
        }

    def consume_stats(self, *, reset: bool = True) -> dict[str, float]:
        self._stats["lookahead_current_occupancy"] = float(len(self._cached_ids))
        stats = dict(self._stats)
        if reset:
            self.reset_stats()
        return stats

    def _resolve_table_name(self) -> str | None:
        eb_configs = getattr(self._embedding_module, "_embedding_bag_configs", None)
        if eb_configs and len(eb_configs) > 0:
            return str(eb_configs[0]["name"])
        return None

    def scan_batches(
        self,
        batch_emb_ids_list: list[set[int]],
    ) -> None:
        """Feed the lookahead window with future batch embedding IDs.

        Called once before training with all batches to build the TTL map.
        Each entry in `batch_emb_ids_list` is the set of unique embedding IDs
        (after fuse, i.e. global fused IDs) for one future batch.

        This implements BagPipe's Oracle Cacher logic:
          TTL[emb_id] = max(batch_idx where emb_id appears)
        """
        self._batch_emb_ids = [set(b) for b in batch_emb_ids_list]
        self._ttl.clear()
        for batch_idx, emb_ids in enumerate(self._batch_emb_ids):
            for eid in emb_ids:
                prev = self._ttl.get(eid)
                if prev is None or batch_idx > prev:
                    self._ttl[eid] = batch_idx

    def _needs_prefetch(self, emb_id: int, current_batch: int) -> bool:
        """Check if an embedding needs to be prefetched for the lookahead window.

        Returns True if the embedding:
          1. Has a TTL >= current_batch (will be used again)
          2. Is not already cached
        """
        ttl = self._ttl.get(emb_id, -1)
        if ttl < current_batch:
            return False
        return emb_id not in self._cached_ids

    def prefetch_and_fill_gpu_cache(
        self,
        current_batch: int,
        fused_ids: list[int],
    ) -> None:
        """Batch prefetch: query needed IDs from PS and fill GPU cache.

        Called at each cleanup_interval step. Implements BagPipe's
        cache refill logic: batch-prefetch the union of all needed IDs
        from the lookahead window, write them into GPU cache.
        """
        if not fused_ids:
            return

        need_prefetch = [
            eid for eid in fused_ids
            if self._needs_prefetch(eid, current_batch)
        ]
        if not need_prefetch:
            return

        prefetch_start = time.perf_counter()
        table_name = self._resolve_table_name()
        if table_name is None:
            return

        ids_tensor = self._torch.tensor(need_prefetch, dtype=self._torch.int64)
        embedding_dim = self._embedding_dim_from_table(table_name)
        if embedding_dim is None:
            return

        prefetch_id = self._kv_client.prefetch(ids_tensor)
        values = self._kv_client.wait_and_get(
            prefetch_id, embedding_dim, device=self._device
        )
        self._kv_client.prefill_gpu_cache(table_name, ids_tensor, values.cpu())

        for eid in need_prefetch:
            self._cached_ids.add(eid)
        self._stats["lookahead_total_prefetched"] += float(len(need_prefetch))
        self._stats["lookahead_prefetch_ms"] += (
            time.perf_counter() - prefetch_start
        ) * 1e3

    def _embedding_dim_from_table(self, table_name: str) -> int | None:
        meta = getattr(self._kv_client, "_tensor_meta", None)
        if meta is None:
            return None
        table_meta = meta.get(table_name)
        if table_meta is None:
            return None
        shape = table_meta.get("shape")
        if shape is None or len(shape) < 2:
            return None
        return int(shape[1])

    def evict_expired(self, current_batch: int) -> None:
        """Evict embeddings whose TTL < current_batch.

        BagPipe-style: embeddings that won't be reused in the remaining
        lookahead window are safe to evict. We simply remove them from
        the tracking set; the GPU cache's LRU will naturally age them out.
        """
        evict_start = time.perf_counter()
        expired = [eid for eid in self._cached_ids if self._ttl.get(eid, -1) < current_batch]
        for eid in expired:
            self._cached_ids.discard(eid)
        self._stats["lookahead_total_evicted"] += float(len(expired))
        self._stats["lookahead_evict_ms"] += (
            time.perf_counter() - evict_start
        ) * 1e3

    def simulate_hit_rate(
        self,
        batch_emb_ids: list[set[int]],
    ) -> dict[str, float]:
        """Offline simulation of hit rate for this lookahead policy.

        Matches the simulation in analyze_lookahead_cache.py.
        Useful for tuning lookahead depth and capacity before deployment.
        """
        ttl: dict[int, int] = {}
        for batch_idx, emb_ids in enumerate(batch_emb_ids):
            for eid in emb_ids:
                ttl[eid] = batch_idx

        cache: dict[int, int] = {}
        total_requests = 0
        total_hits = 0
        prefetch_ids = 0
        cache_insert_ids = 0
        evicted = 0

        for batch_idx, current in enumerate(batch_emb_ids):
            expired = [eid for eid, last in cache.items() if last < batch_idx]
            for eid in expired:
                del cache[eid]
                evicted += 1

            future_use = {eid for eid in current if ttl.get(eid, -1) >= batch_idx}
            cache_hits = current & set(cache)
            total_hits += len(cache_hits)
            total_requests += len(current)

            for eid in future_use:
                if eid not in cache:
                    if len(cache) >= self._capacity:
                        victim = min(cache.items(), key=lambda item: item[1])[0]
                        del cache[victim]
                        evicted += 1
                    if len(cache) < self._capacity:
                        cache[eid] = ttl[eid]
                        cache_insert_ids += 1

            prefetch_ids += len(current - cache_hits)

        misses = total_requests - total_hits
        return {
            "policy": "lookahead_ttl",
            "capacity": self._capacity,
            "depth": self._lookahead,
            "requests": total_requests,
            "hits": total_hits,
            "misses": misses,
            "hit_rate": total_hits / total_requests if total_requests else 0.0,
            "prefetch_ids": prefetch_ids,
            "cache_insert_ids": cache_insert_ids,
            "evicted_ids": evicted,
        }

    def on_batch_start(self, batch_idx: int, fused_ids: set[int]) -> None:
        """Called at the start of each training batch.

        At cleanup intervals:
          1. Evict expired embeddings from tracking
          2. Prefetch future-needed embeddings and fill GPU cache
        """
        if batch_idx % self._cleanup_interval != 0:
            return
        self.evict_expired(batch_idx)
        if batch_idx + self._lookahead < len(self._batch_emb_ids):
            future_window = set().union(*self._batch_emb_ids[batch_idx:batch_idx + self._lookahead])
            self.prefetch_and_fill_gpu_cache(batch_idx, list(future_window))

    def on_batch_end(self, batch_idx: int) -> None:
        """Called at the end of each training batch.

        Currently a no-op; future work could add async gradient sync
        overlap similar to BagPipe's sync_now/sync_later split.
        """
        pass

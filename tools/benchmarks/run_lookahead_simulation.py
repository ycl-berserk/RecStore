#!/usr/bin/env python3
"""
Offline simulation of Lookahead TTL cache vs LRU cache for RecStore GPU cache.

This script does NOT require PyTorch. It uses synthetic data with configurable
power-law skew to simulate embedding access patterns typical in DLRM workloads.

Usage:
    python tools/benchmarks/run_lookahead_simulation.py \
        --num-embeddings 100000 --batch-size 4096 --steps 500 \
        --depth 200 --capacity 50000

Outputs CSV files with hit rate comparisons.
"""

import argparse
import csv
import math
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


def generate_synthetic_batches(
    num_embeddings: int,
    num_tables: int,
    batch_size: int,
    steps: int,
    seed: int,
    alpha: float = 0.8,
) -> list[set[int]]:
    """Generate synthetic fused embedding IDs with power-law skew.

    Args:
        alpha: Power-law exponent. Lower = more skew (hotter head).
               Typical DLRM workloads have alpha ~ 0.7-0.9.
    """
    rng = random.Random(seed)

    # Fused ID range: each table gets its own offset
    table_offsets = [num_embeddings * i for i in range(num_tables)]

    def power_law_sample(max_id: int) -> int:
        # Zipf-like: lower IDs are more frequent
        return int(max_id * (1.0 - (1.0 - rng.random()) ** (1.0 / alpha))) % max_id

    batches: list[set[int]] = []
    for _ in range(steps):
        ids: set[int] = set()
        # Each batch touches many tables (typical DLRM: all 26 tables per batch)
        for table_idx in range(num_tables):
            offset = table_offsets[table_idx]
            # Each table contributes some unique IDs per batch
            avg_ids_per_table = max(1, batch_size // num_tables)
            for _ in range(avg_ids_per_table):
                local_id = power_law_sample(num_embeddings)
                fused = offset + local_id
                ids.add(fused)
        batches.append(ids)
    return batches


def simulate_lru(
    batches: list[set[int]],
    *,
    capacity: int,
    bypass_min_rows: int | None = None,
    low_hit_ratio: float = 0.05,
) -> dict[str, Any]:
    cache: OrderedDict[int, None] = OrderedDict()
    total_hits = 0
    total_requests = 0
    total_misses = 0
    bypassed_batches = 0
    queried_batches = 0
    bypass_enabled = False

    for batch in batches:
        ids = batch
        total_requests += len(ids)
        if bypass_min_rows is not None and bypass_enabled and len(ids) >= bypass_min_rows:
            total_misses += len(ids)
            bypassed_batches += 1
            continue
        hits = sum(1 for emb_id in ids if emb_id in cache)
        total_hits += hits
        misses = len(ids) - hits
        total_misses += misses
        queried_batches += 1
        for emb_id in ids:
            if emb_id in cache:
                cache.move_to_end(emb_id)
            else:
                cache[emb_id] = None
                while len(cache) > capacity:
                    cache.popitem(last=False)
        if (
            bypass_min_rows is not None
            and len(ids) >= bypass_min_rows
            and len(ids) > 0
            and hits / len(ids) < low_hit_ratio
        ):
            bypass_enabled = True
            cache = OrderedDict()

    return {
        "policy": "lru_bypass" if bypass_min_rows is not None else "lru_all_insert",
        "capacity": capacity,
        "depth": 0,
        "requests": total_requests,
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total_requests if total_requests else 0.0,
        "bypassed_batches": bypassed_batches,
        "prefetch_ids": total_misses,
        "cache_insert_ids": total_misses,
    }


def simulate_lookahead_ttl(
    batches: list[set[int]],
    *,
    capacity: int,
    depth: int,
) -> dict[str, Any]:
    cache: dict[int, int] = {}
    evicted = 0
    hits = 0
    requests = 0
    prefetch_ids = 0
    cache_insert_ids = 0

    for idx, current in enumerate(batches):
        expired = [eid for eid, ttl in cache.items() if ttl < idx]
        for eid in expired:
            del cache[eid]
            evicted += 1

        future_batches = batches[idx + 1 : idx + depth + 1]
        last_use: dict[int, int] = {}
        for offset, future in enumerate(future_batches, start=1):
            for emb_id in future:
                last_use[emb_id] = idx + offset

        reusable = current & set(last_use)
        cache_hits = current & set(cache)
        hits += len(cache_hits)
        requests += len(current)

        for emb_id in reusable:
            ttl = last_use[emb_id]
            if emb_id not in cache:
                if len(cache) >= capacity:
                    victim = min(cache.items(), key=lambda item: item[1])[0]
                    del cache[victim]
                    evicted += 1
                if len(cache) < capacity:
                    cache[emb_id] = ttl
                    cache_insert_ids += 1
            else:
                cache[emb_id] = max(cache[emb_id], ttl)

        prefetch_ids += len(current - cache_hits)

    misses = requests - hits
    return {
        "policy": "lookahead_ttl",
        "capacity": capacity,
        "depth": depth,
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / requests if requests else 0.0,
        "prefetch_ids": prefetch_ids,
        "cache_insert_ids": cache_insert_ids,
        "evicted_ids": evicted,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cleaned = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline simulation of LRU vs Lookahead TTL cache policies."
    )
    parser.add_argument("--num-embeddings", type=int, default=100000)
    parser.add_argument("--num-tables", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.8, help="Power-law skew (lower = hotter)")
    parser.add_argument("--depths", type=str, default="1,2,4,8,16,32,64,128,200")
    parser.add_argument("--capacities", type=str, default="1024,4096,16384,50000")
    parser.add_argument("--output", type=str, default="/tmp/recstore_lookahead_sim.csv")
    args = parser.parse_args()

    depths = [int(v) for v in args.depths.split(",")]
    capacities = [int(v) for v in args.capacities.split(",")]

    print(f"Generating {args.steps} synthetic batches with {args.num_embeddings} embeddings...")
    batches = generate_synthetic_batches(
        num_embeddings=args.num_embeddings,
        num_tables=args.num_tables,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        alpha=args.alpha,
    )
    total_unique = len(set().union(*batches))
    print(f"Generated {len(batches)} batches, {total_unique} unique fused IDs")

    rows: list[dict[str, Any]] = []
    for capacity in capacities:
        print(f"\n--- capacity={capacity} ---")

        lru = simulate_lru(batches, capacity=capacity)
        rows.append(lru)
        print(f"  LRU:          hit_rate={lru['hit_rate']:.4f}")

        lru_bp = simulate_lru(batches, capacity=capacity, bypass_min_rows=1024)
        rows.append(lru_bp)
        print(f"  LRU+bypass:   hit_rate={lru_bp['hit_rate']:.4f} (bypassed={lru_bp['bypassed_batches']})")

        for depth in depths:
            la = simulate_lookahead_ttl(batches, capacity=capacity, depth=depth)
            rows.append(la)
            improvement = (la['hit_rate'] - lru['hit_rate']) / lru['hit_rate'] * 100 if lru['hit_rate'] > 0 else 0
            print(f"  Lookahead d={depth:3d}: hit_rate={la['hit_rate']:.4f}  (+{improvement:+.1f}% vs LRU)")

    out_path = Path(args.output)
    write_csv(out_path, rows)
    print(f"\nResults written to {out_path}")

    best = max(rows, key=lambda r: r['hit_rate'])
    print(f"\n=== Best policy ===")
    print(f"  {best['policy']} (capacity={best['capacity']}, depth={best['depth']})")
    print(f"  hit_rate={best['hit_rate']:.4f}")


if __name__ == "__main__":
    main()

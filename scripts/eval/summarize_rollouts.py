#!/usr/bin/env python3
"""Summarize local MuJoCo rollout JSONL without uploading raw data.

Expected one JSON object per episode:
{"policy":"act", "split":"id", "success":true, "started":true,
 "grasped":true, "placed":true, "completion_time_s":13.2,
 "inference_ms":[22.4, 19.0], "seed":42}

This is an aggregator ONLY, not an environment rollout runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * percent / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] * (hi - pos) + data[hi] * (pos - lo) if lo != hi else data[lo]


def summarize(events: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for idx, event in enumerate(events, 1):
        if not isinstance(event, dict) or not isinstance(event.get("policy"), str) or not event["policy"].strip() or not isinstance(event.get("success"), bool):
            raise ValueError(f"line {idx}: policy (string) and success (boolean) are required")
        if "split" in event and not isinstance(event["split"], str):
            raise ValueError(f"line {idx}: split must be a string")
        for key in ("started", "grasped", "placed"):
            if key in event and not isinstance(event[key], bool):
                raise ValueError(f"line {idx}: {key} must be a boolean")
        numbers = []
        if "completion_time_s" in event:
            numbers.append(event["completion_time_s"])
        if "inference_ms" in event:
            if not isinstance(event["inference_ms"], list):
                raise ValueError(f"line {idx}: inference_ms must be a list")
            numbers.extend(event["inference_ms"])
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in numbers):
            raise ValueError(f"line {idx}: times must be finite nonnegative numbers")
        if "completed" in event and not isinstance(event["completed"], bool):
            raise ValueError(f"line {idx}: completed must be a boolean")
        if event.get("completed", True):
            groups[(event["policy"], event.get("split", "all"))].append(event)

    output = []
    for (policy, split), items in sorted(groups.items()):
        n = len(items)
        wins = sum(e["success"] for e in items)
        lower, upper = wilson_interval(wins, n)
        complete = [float(e["completion_time_s"]) for e in items if e["success"] and isinstance(e.get("completion_time_s"), (int, float))]
        latencies = [float(x) for e in items for x in e.get("inference_ms", []) if isinstance(x, (int, float))]
        started = [e["started"] for e in items if isinstance(e.get("started"), bool)]
        grasped = [e["grasped"] for e in items if isinstance(e.get("grasped"), bool)]
        placed = [e["placed"] for e in items if isinstance(e.get("placed"), bool)]
        output.append({
            "policy": policy,
            "split": split,
            "episodes": n,
            "successes": wins,
            "success_rate_pct": round(100 * wins / n, 2),
            "wilson95_lower_pct": round(100 * lower, 2),
            "wilson95_upper_pct": round(100 * upper, 2),
            "avg_success_time_s": round(statistics.mean(complete), 3) if complete else None,
            "p50_inference_ms": round(percentile(latencies, 50), 3) if latencies else None,
            "p95_inference_ms": round(percentile(latencies, 95), 3) if latencies else None,
            "startup_fail_pct": round(100 * (len(started) - sum(started)) / len(started), 2) if len(started) == n else None,
            "grasp_fail_pct_all": round(100 * (len(grasped) - sum(grasped)) / len(grasped), 2) if len(grasped) == n else None,
            "place_fail_pct_all": round(100 * (len(placed) - sum(placed)) / len(placed), 2) if len(placed) == n else None,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local rollout JSONL file")
    parser.add_argument("--csv", type=Path, help="Save anonymous aggregate CSV (opt-in)")
    args = parser.parse_args()
    events = []
    with args.input.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON on line {line_num}: {exc}") from exc
    if not events:
        raise SystemExit("No episodes found")
    result = summarize(events)
    if not result:
        raise SystemExit("No completed episodes found")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result[0].keys())
            writer.writeheader()
            writer.writerows(result)
        print(f"Saved aggregate CSV: {args.csv}")


if __name__ == "__main__":
    main()

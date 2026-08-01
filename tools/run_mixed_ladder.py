#!/usr/bin/env python3
"""Sweep body/head fractional-bit combinations on the tiny MGW oracle."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mgw_precision import (
    PrecisionError,
    quantize_mgw_plan,
    sha256,
    verify_mgw_plan,
)
from run_ladder import compare, run_model


BODY_BITS = (12, 11, 10, 9, 8, 7, 6)
HEAD_BITS = (9, 10, 11, 12, 13, 14, 16, 24, 48)
EXACT_METADATA = {
    "tokenizer.uchars": 48,
    "rng.state": 48,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    model = args.model.resolve()
    baseline = run_model(binary, model)
    results: list[dict[str, object]] = []

    print(
        "body head samples trace-token top1 first-token first-top1"
    )
    with tempfile.TemporaryDirectory(prefix="mgpt-mixed-") as directory:
        temp_root = Path(directory)
        for body_bits in BODY_BITS:
            for head_bits in HEAD_BITS:
                if head_bits < body_bits:
                    continue
                plan = {
                    **EXACT_METADATA,
                    "lm_head": head_bits,
                }
                candidate = (
                    temp_root
                    / f"model-body-f{body_bits:02d}-head-f{head_bits:02d}.mgw"
                )
                by_tensor = quantize_mgw_plan(
                    model,
                    candidate,
                    body_bits,
                    plan,
                )
                verify_mgw_plan(
                    model,
                    candidate,
                    body_bits,
                    plan,
                )
                output = run_model(binary, candidate)
                total_weights = sum(
                    stats.total_weights
                    for name, stats in by_tensor.items()
                    if name not in EXACT_METADATA
                )
                changed_weights = sum(
                    stats.changed_weights
                    for name, stats in by_tensor.items()
                    if name not in EXACT_METADATA
                )
                row = {
                    "body_fraction_bits": body_bits,
                    "head_fraction_bits": head_bits,
                    "total_weights": total_weights,
                    "changed_weights": changed_weights,
                    **compare(
                        baseline,
                        output,
                        by_tensor["lm_head"],
                        sha256(candidate),
                    ),
                }
                results.append(row)
                print(
                    f"{body_bits:4d} "
                    f"{head_bits:4d} "
                    f"{int(row['matching_samples']):2d}/20 "
                    f"{int(row['matching_trace_tokens']):3d}/"
                    f"{int(row['common_trace_points']):3d} "
                    f"{int(row['matching_top1']):3d}/"
                    f"{int(row['common_trace_points']):3d} "
                    f"{str(row['first_output_token_divergence']) or '-':>11s} "
                    f"{str(row['first_top1_divergence']) or '-':>10s}"
                )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "model_sha256": sha256(model),
        "binary": str(args.binary),
        "body_fraction_bits": list(BODY_BITS),
        "head_fraction_bits": list(HEAD_BITS),
        "exact_metadata": sorted(EXACT_METADATA),
        "baseline_sample_sha256": baseline.sample_hash,
        "baseline_trace_sha256": baseline.trace_hash,
        "baseline_trace_points": len(baseline.trace),
        "results": results,
    }
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nCSV:  {args.csv}")
    print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PrecisionError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

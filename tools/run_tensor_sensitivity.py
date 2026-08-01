#!/usr/bin/env python3
"""Probe which tiny-model tensor groups control the precision boundary."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mgw_precision import (
    PrecisionError,
    parse_mgw,
    quantize_mgw,
    sha256,
    verify_candidate,
)
from run_ladder import compare, parse_bits, run_model


GROUPS: dict[str, frozenset[str]] = {
    "token_embedding": frozenset({"wte"}),
    "position_embedding": frozenset({"wpe"}),
    "language_head": frozenset({"lm_head"}),
    "attention_qkv": frozenset(
        {
            "layers.0.attn_wq",
            "layers.0.attn_wk",
            "layers.0.attn_wv",
        }
    ),
    "attention_output": frozenset({"layers.0.attn_wo"}),
    "mlp_fc1": frozenset({"layers.0.mlp_fc1"}),
    "mlp_fc2": frozenset({"layers.0.mlp_fc2"}),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bits", default="11,10,9")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    model = args.model.resolve()
    info = parse_mgw(model)
    all_weights = frozenset(
        tensor.name for tensor in info.tensors if tensor.is_weight
    )
    grouped_weights = frozenset().union(*GROUPS.values())
    if grouped_weights != all_weights:
        raise PrecisionError(
            "tensor groups do not exactly cover model weights: "
            f"missing={sorted(all_weights - grouped_weights)}, "
            f"extra={sorted(grouped_weights - all_weights)}"
        )

    baseline = run_model(binary, model)
    results: list[dict[str, object]] = []
    scenarios: list[tuple[str, str, frozenset[str]]] = [
        ("all", "all tensors rounded", all_weights)
    ]
    for group, names in GROUPS.items():
        scenarios.append(("only", group, names))
    for group, names in GROUPS.items():
        scenarios.append(("rescue", group, all_weights - names))

    print(
        " F mode   group                 weights samples trace-token top1 "
        "first-token first-top1"
    )
    with tempfile.TemporaryDirectory(prefix="mgpt-tensor-") as directory:
        temp_root = Path(directory)
        for bits in parse_bits(args.bits):
            for scenario_index, (mode, group, selected) in enumerate(scenarios):
                candidate = (
                    temp_root
                    / f"model-f{bits:02d}-{scenario_index:02d}.mgw"
                )
                stats = quantize_mgw(
                    model,
                    candidate,
                    bits,
                    tensor_names=selected,
                )
                verify_candidate(
                    model,
                    candidate,
                    bits,
                    tensor_names=selected,
                )
                output = run_model(binary, candidate)
                row = {
                    "mode": mode,
                    "group": group,
                    "rounded_tensors": ",".join(sorted(selected)),
                    **compare(
                        baseline,
                        output,
                        stats,
                        sha256(candidate),
                    ),
                }
                results.append(row)
                print(
                    f"{bits:2d} "
                    f"{mode:6s} "
                    f"{group:21s} "
                    f"{stats.total_weights:7d} "
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
        "fraction_bits": parse_bits(args.bits),
        "groups": {
            group: sorted(tensors) for group, tensors in GROUPS.items()
        },
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

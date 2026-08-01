#!/usr/bin/env python3
"""Run a weight-only fractional-bit sweep against the tiny MicroGPT oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from mgw_precision import (
    PrecisionError,
    QuantizationStats,
    parse_mgw,
    quantize_mgw,
    sha256,
    verify_candidate,
)


TRACE_RE = re.compile(
    r"^TRACE s=(?P<sample>\d+) p=(?P<position>\d+) "
    r"in=(?P<input>-?\d+) out=(?P<output>-?\d+) "
    r"lh=(?P<logits_hash>[0-9a-f]{16}) "
    r"ph=(?P<probs_hash>[0-9a-f]{16}) "
    r"top1=(?P<top1>-?\d+) topv=(?P<top_value>-?\d+) "
    r"margin=(?P<margin>-?\d+)$"
)
SAMPLE_RE = re.compile(r"^sample\s+(?P<number>\d+):(?: )?(?P<text>.*)$")


@dataclass(frozen=True)
class TracePoint:
    sample: int
    position: int
    input: int
    output: int
    logits_hash: str
    probs_hash: str
    top1: int
    top_value: int
    margin: int

    @property
    def key(self) -> str:
        return f"{self.sample}:{self.position}"


@dataclass(frozen=True)
class RunOutput:
    samples: tuple[str, ...]
    trace: tuple[TracePoint, ...]
    sample_hash: str
    trace_hash: str


def canonical_hash(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def run_model(binary: Path, model: Path) -> RunOutput:
    completed = subprocess.run(
        [str(binary), "--load", str(model)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise PrecisionError(
            f"{binary} failed for {model} with {completed.returncode}:\n"
            f"{completed.stdout}"
        )
    samples: list[str] = []
    trace: list[TracePoint] = []
    trace_lines: list[str] = []
    for line in completed.stdout.splitlines():
        if match := TRACE_RE.match(line):
            values = match.groupdict()
            trace.append(
                TracePoint(
                    sample=int(values["sample"]),
                    position=int(values["position"]),
                    input=int(values["input"]),
                    output=int(values["output"]),
                    logits_hash=values["logits_hash"],
                    probs_hash=values["probs_hash"],
                    top1=int(values["top1"]),
                    top_value=int(values["top_value"]),
                    margin=int(values["margin"]),
                )
            )
            trace_lines.append(line)
        elif match := SAMPLE_RE.match(line):
            samples.append(match.group("text"))
    if len(samples) != 20:
        raise PrecisionError(
            f"expected 20 samples from {model}, received {len(samples)}"
        )
    if not trace:
        raise PrecisionError(f"no MGPT_PRECISION_TRACE output from {binary}")
    return RunOutput(
        tuple(samples),
        tuple(trace),
        canonical_hash([f"{index + 1}:{text}" for index, text in enumerate(samples)]),
        canonical_hash(trace_lines),
    )


def location(point: TracePoint | None) -> str:
    return point.key if point is not None else ""


def first_difference(
    baseline: tuple[TracePoint, ...],
    candidate: tuple[TracePoint, ...],
    field: str,
) -> TracePoint | None:
    candidate_by_key = {point.key: point for point in candidate}
    for point in baseline:
        other = candidate_by_key.get(point.key)
        if other is None or getattr(point, field) != getattr(other, field):
            return point
    if len(candidate) != len(baseline):
        return candidate[len(baseline)] if len(candidate) > len(baseline) else baseline[-1]
    return None


def first_sample_difference(
    baseline: tuple[str, ...],
    candidate: tuple[str, ...],
) -> tuple[int | None, int | None]:
    for sample_index, (expected, actual) in enumerate(zip(baseline, candidate)):
        if expected == actual:
            continue
        limit = min(len(expected), len(actual))
        char_index = next(
            (index for index in range(limit) if expected[index] != actual[index]),
            limit,
        )
        return sample_index, char_index
    return None, None


def compare(
    baseline: RunOutput,
    candidate: RunOutput,
    stats: QuantizationStats,
    model_hash: str,
) -> dict[str, object]:
    baseline_by_key = {point.key: point for point in baseline.trace}
    candidate_by_key = {point.key: point for point in candidate.trace}
    common_keys = baseline_by_key.keys() & candidate_by_key.keys()
    matching_trace_tokens = sum(
        baseline_by_key[key].output == candidate_by_key[key].output
        for key in common_keys
    )
    matching_top1 = sum(
        baseline_by_key[key].top1 == candidate_by_key[key].top1
        for key in common_keys
    )
    max_top_value_delta = max(
        (
            abs(
                baseline_by_key[key].top_value
                - candidate_by_key[key].top_value
            )
            for key in common_keys
        ),
        default=0,
    )
    sample_index, char_index = first_sample_difference(
        baseline.samples, candidate.samples
    )
    logits_diff = first_difference(baseline.trace, candidate.trace, "logits_hash")
    probs_diff = first_difference(baseline.trace, candidate.trace, "probs_hash")
    output_diff = first_difference(baseline.trace, candidate.trace, "output")
    top1_diff = first_difference(baseline.trace, candidate.trace, "top1")
    result: dict[str, object] = {
        **stats.to_dict(),
        "candidate_sha256": model_hash,
        "sample_sha256": candidate.sample_hash,
        "trace_sha256": candidate.trace_hash,
        "matching_samples": sum(
            expected == actual
            for expected, actual in zip(baseline.samples, candidate.samples)
        ),
        "first_sample_difference": (
            "" if sample_index is None else f"{sample_index}:{char_index}"
        ),
        "trace_points": len(candidate.trace),
        "matching_trace_tokens": matching_trace_tokens,
        "common_trace_points": len(common_keys),
        "matching_top1": matching_top1,
        "max_abs_top1_value_delta_raw": max_top_value_delta,
        "first_logits_divergence": location(logits_diff),
        "first_probs_divergence": location(probs_diff),
        "first_output_token_divergence": location(output_diff),
        "first_top1_divergence": location(top1_diff),
    }
    return result


def parse_bits(specification: str) -> list[int]:
    if specification == "all":
        return list(range(48, -1, -1))
    result: list[int] = []
    for item in specification.split(","):
        value = int(item)
        if not 0 <= value <= 48:
            raise PrecisionError(f"fraction bit {value} outside [0, 48]")
        if value not in result:
            result.append(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bits", default="all")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    model = args.model.resolve()
    parse_mgw(model)
    fraction_bits = parse_bits(args.bits)
    baseline = run_model(binary, model)
    baseline_model_hash = sha256(model)
    results: list[dict[str, object]] = []

    print(
        " F  changed  zeroed samples trace-token top1 "
        "first-logit first-token"
    )
    with tempfile.TemporaryDirectory(prefix="mgpt-precision-") as directory:
        temp_root = Path(directory)
        for bits in fraction_bits:
            candidate = temp_root / f"model-f{bits:02d}.mgw"
            stats = quantize_mgw(model, candidate, bits)
            verify_candidate(model, candidate, bits)
            output = run_model(binary, candidate)
            row = compare(baseline, output, stats, sha256(candidate))
            results.append(row)
            print(
                f"{bits:2d} "
                f"{stats.changed_weights:8d} "
                f"{stats.quantized_zeros - stats.original_zeros:7d} "
                f"{int(row['matching_samples']):2d}/20 "
                f"{int(row['matching_trace_tokens']):3d}/"
                f"{int(row['common_trace_points']):3d} "
                f"{int(row['matching_top1']):3d}/"
                f"{int(row['common_trace_points']):3d} "
                f"{str(row['first_logits_divergence']) or '-':>11s} "
                f"{str(row['first_output_token_divergence']) or '-':>11s}"
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
        "model_sha256": baseline_model_hash,
        "binary": str(args.binary),
        "baseline_sample_sha256": baseline.sample_hash,
        "baseline_trace_sha256": baseline.trace_hash,
        "baseline_samples": list(baseline.samples),
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

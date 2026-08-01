#!/usr/bin/env python3
"""Deterministic fractional-precision experiments for MicroGPT MGW files."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path


FP_PRECISION = 48
MGW_MAGIC = b"MGW\0"
MGW_VERSION = 1
MGW_ENDIAN_TAG = 0x01020304
HEADER = struct.Struct("<4sIIIQQ32s")
CONFIG = struct.Struct("<16i")
INDEX = struct.Struct("<64sQQIIII")
NON_WEIGHT_TENSORS = {"tokenizer.uchars", "rng.state"}
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


class PrecisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tensor:
    name: str
    num_elements: int
    data_offset: int
    ndims: int
    shape0: int
    shape1: int

    @property
    def byte_size(self) -> int:
        return self.num_elements * 8

    @property
    def is_weight(self) -> bool:
        return self.name not in NON_WEIGHT_TENSORS


@dataclass(frozen=True)
class ModelInfo:
    path: str
    size: int
    vocab_size: int
    hidden_dim: int
    num_layers: int
    tensors: tuple[Tensor, ...]


@dataclass
class QuantizationStats:
    fraction_bits: int
    dropped_bits: int
    total_weights: int = 0
    changed_weights: int = 0
    original_zeros: int = 0
    quantized_zeros: int = 0
    max_abs_delta_raw: int = 0
    sum_abs_delta_raw: int = 0
    saturations: int = 0

    @property
    def mean_abs_delta_raw(self) -> float:
        if self.total_weights == 0:
            return 0.0
        return self.sum_abs_delta_raw / self.total_weights

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["mean_abs_delta_raw"] = self.mean_abs_delta_raw
        return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mgw(path: Path) -> ModelInfo:
    raw = path.read_bytes()
    if len(raw) < HEADER.size + CONFIG.size:
        raise PrecisionError(f"{path}: file is too small")
    (
        magic,
        version,
        endian_tag,
        num_tensors,
        index_offset,
        _data_offset,
        _reserved,
    ) = HEADER.unpack_from(raw, 0)
    if magic != MGW_MAGIC or version != MGW_VERSION:
        raise PrecisionError(f"{path}: not an MGW v1 file")
    if endian_tag != MGW_ENDIAN_TAG:
        raise PrecisionError(f"{path}: this spike expects little-endian MGW")
    config = CONFIG.unpack_from(raw, HEADER.size)
    hidden_dim = config[0]
    num_layers = config[4]
    vocab_size = config[6]
    index_size = num_tensors * INDEX.size
    if index_offset > len(raw) or index_size > len(raw) - index_offset:
        raise PrecisionError(f"{path}: tensor index is out of bounds")

    tensors: list[Tensor] = []
    names: set[str] = set()
    for tensor_index in range(num_tensors):
        offset = index_offset + tensor_index * INDEX.size
        (
            raw_name,
            num_elements,
            data_offset,
            ndims,
            shape0,
            shape1,
            _tensor_reserved,
        ) = INDEX.unpack_from(raw, offset)
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        if not name or name in names:
            raise PrecisionError(f"{path}: invalid/duplicate tensor name {name!r}")
        names.add(name)
        expected = shape0 if ndims == 1 else shape0 * shape1
        if ndims not in (1, 2) or num_elements != expected:
            raise PrecisionError(f"{path}: invalid shape for tensor {name}")
        byte_size = num_elements * 8
        if data_offset > len(raw) or byte_size > len(raw) - data_offset:
            raise PrecisionError(f"{path}: tensor {name} is out of bounds")
        tensors.append(
            Tensor(name, num_elements, data_offset, ndims, shape0, shape1)
        )

    missing = NON_WEIGHT_TENSORS - names
    if missing:
        raise PrecisionError(f"{path}: missing metadata tensors {sorted(missing)}")
    return ModelInfo(
        str(path),
        len(raw),
        vocab_size,
        hidden_dim,
        num_layers,
        tuple(tensors),
    )


def round_q1648(value: int, fraction_bits: int) -> tuple[int, bool]:
    """Round to a coarser binary grid; halfway cases go away from zero."""
    if not 0 <= fraction_bits <= FP_PRECISION:
        raise PrecisionError("fraction_bits must be in [0, 48]")
    dropped = FP_PRECISION - fraction_bits
    if dropped == 0:
        return value, False
    quantum = 1 << dropped
    half = quantum >> 1
    magnitude = -value if value < 0 else value
    rounded_magnitude = ((magnitude + half) // quantum) * quantum
    rounded = -rounded_magnitude if value < 0 else rounded_magnitude
    saturated = False
    if rounded < INT64_MIN:
        rounded = INT64_MIN
        saturated = True
    elif rounded > INT64_MAX:
        rounded = INT64_MAX
        saturated = True
    return rounded, saturated


def selected_weight_names(
    info: ModelInfo,
    tensor_names: set[str] | frozenset[str] | None,
) -> frozenset[str]:
    weights = frozenset(tensor.name for tensor in info.tensors if tensor.is_weight)
    if tensor_names is None:
        return weights
    selected = frozenset(tensor_names)
    invalid = selected - weights
    if invalid:
        raise PrecisionError(
            f"not selectable weight tensors: {sorted(invalid)}"
        )
    return selected


def quantize_mgw(
    source: Path,
    destination: Path,
    fraction_bits: int,
    tensor_names: set[str] | frozenset[str] | None = None,
) -> QuantizationStats:
    if destination.exists():
        raise PrecisionError(f"refusing to overwrite {destination}")
    info = parse_mgw(source)
    selected = selected_weight_names(info, tensor_names)
    raw = bytearray(source.read_bytes())
    stats = QuantizationStats(fraction_bits, FP_PRECISION - fraction_bits)

    for tensor in info.tensors:
        if tensor.name not in selected:
            continue
        for element in range(tensor.num_elements):
            offset = tensor.data_offset + element * 8
            (value,) = struct.unpack_from("<q", raw, offset)
            quantized, saturated = round_q1648(value, fraction_bits)
            delta = abs(quantized - value)
            stats.total_weights += 1
            stats.changed_weights += quantized != value
            stats.original_zeros += value == 0
            stats.quantized_zeros += quantized == 0
            stats.max_abs_delta_raw = max(stats.max_abs_delta_raw, delta)
            stats.sum_abs_delta_raw += delta
            stats.saturations += saturated
            struct.pack_into("<q", raw, offset, quantized)

    destination.write_bytes(raw)
    return stats


def quantize_mgw_plan(
    source: Path,
    destination: Path,
    default_fraction_bits: int,
    tensor_fraction_bits: dict[str, int],
) -> dict[str, QuantizationStats]:
    """Apply one fractional-bit setting per tensor.

    This in-memory reference implementation is intended for the tiny oracle.
    The C streaming converter is used for multi-gigabyte models and is tested
    byte-for-byte against this function.
    """
    if destination.exists():
        raise PrecisionError(f"refusing to overwrite {destination}")
    if not 0 <= default_fraction_bits <= FP_PRECISION:
        raise PrecisionError("default_fraction_bits must be in [0, 48]")
    info = parse_mgw(source)
    names = {tensor.name for tensor in info.tensors}
    invalid = set(tensor_fraction_bits) - names
    if invalid:
        raise PrecisionError(f"unknown tensor names: {sorted(invalid)}")
    for name, bits in tensor_fraction_bits.items():
        if not 0 <= bits <= FP_PRECISION:
            raise PrecisionError(
                f"fraction bits for {name} must be in [0, 48]"
            )

    raw = bytearray(source.read_bytes())
    stats_by_tensor: dict[str, QuantizationStats] = {}
    for tensor in info.tensors:
        bits = tensor_fraction_bits.get(tensor.name, default_fraction_bits)
        stats = QuantizationStats(bits, FP_PRECISION - bits)
        stats_by_tensor[tensor.name] = stats
        for element in range(tensor.num_elements):
            offset = tensor.data_offset + element * 8
            (value,) = struct.unpack_from("<q", raw, offset)
            quantized, saturated = round_q1648(value, bits)
            delta = abs(quantized - value)
            stats.total_weights += 1
            stats.changed_weights += quantized != value
            stats.original_zeros += value == 0
            stats.quantized_zeros += quantized == 0
            stats.max_abs_delta_raw = max(stats.max_abs_delta_raw, delta)
            stats.sum_abs_delta_raw += delta
            stats.saturations += saturated
            struct.pack_into("<q", raw, offset, quantized)

    destination.write_bytes(raw)
    return stats_by_tensor


def verify_mgw_plan(
    source: Path,
    candidate: Path,
    default_fraction_bits: int,
    tensor_fraction_bits: dict[str, int],
) -> None:
    source_info = parse_mgw(source)
    candidate_info = parse_mgw(candidate)
    if source_info.size != candidate_info.size:
        raise PrecisionError("candidate size differs")
    if source_info.tensors != candidate_info.tensors:
        raise PrecisionError("candidate tensor index differs")
    source_raw = source.read_bytes()
    candidate_raw = candidate.read_bytes()
    for tensor in source_info.tensors:
        bits = tensor_fraction_bits.get(
            tensor.name,
            default_fraction_bits,
        )
        quantum = 1 << (FP_PRECISION - bits)
        region = candidate_raw[
            tensor.data_offset : tensor.data_offset + tensor.byte_size
        ]
        for (value,) in struct.iter_unpack("<q", region):
            if value % quantum:
                raise PrecisionError(
                    f"tensor {tensor.name} contains off-grid value {value}"
                )
        if bits == FP_PRECISION:
            source_region = source_raw[
                tensor.data_offset : tensor.data_offset + tensor.byte_size
            ]
            if region != source_region:
                raise PrecisionError(f"F48 tensor {tensor.name} changed")


def verify_candidate(
    source: Path,
    candidate: Path,
    fraction_bits: int,
    tensor_names: set[str] | frozenset[str] | None = None,
) -> None:
    source_info = parse_mgw(source)
    candidate_info = parse_mgw(candidate)
    selected = selected_weight_names(source_info, tensor_names)
    if source_info.size != candidate_info.size:
        raise PrecisionError("candidate size differs")
    if source_info.tensors != candidate_info.tensors:
        raise PrecisionError("candidate tensor index differs")
    source_raw = source.read_bytes()
    candidate_raw = candidate.read_bytes()
    dropped = FP_PRECISION - fraction_bits
    quantum = 1 << dropped
    for tensor in source_info.tensors:
        source_region = source_raw[
            tensor.data_offset : tensor.data_offset + tensor.byte_size
        ]
        candidate_region = candidate_raw[
            tensor.data_offset : tensor.data_offset + tensor.byte_size
        ]
        if tensor.name not in selected:
            if source_region != candidate_region:
                raise PrecisionError(f"unselected tensor {tensor.name} changed")
            continue
        for (value,) in struct.iter_unpack("<q", candidate_region):
            if value % quantum:
                raise PrecisionError(
                    f"tensor {tensor.name} contains off-grid value {value}"
                )


def print_info(info: ModelInfo) -> None:
    print(f"path:        {info.path}")
    print(f"bytes:       {info.size:,}")
    print(f"vocab:       {info.vocab_size}")
    print(f"hidden:      {info.hidden_dim}")
    print(f"layers:      {info.num_layers}")
    print("tensors:")
    for tensor in info.tensors:
        kind = "weight" if tensor.is_weight else "metadata"
        shape = (
            f"[{tensor.shape0}]"
            if tensor.ndims == 1
            else f"[{tensor.shape0}, {tensor.shape1}]"
        )
        print(
            f"  {tensor.name:24s} {shape:12s} {kind:8s} "
            f"{tensor.byte_size:8,d} bytes"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("model", type=Path)

    quantize_parser = subparsers.add_parser("quantize")
    quantize_parser.add_argument("--input", type=Path, required=True)
    quantize_parser.add_argument("--output", type=Path, required=True)
    quantize_parser.add_argument("--fraction-bits", type=int, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--candidate", type=Path, required=True)
    verify_parser.add_argument("--fraction-bits", type=int, required=True)

    args = parser.parse_args()
    if args.command == "inspect":
        print_info(parse_mgw(args.model))
    elif args.command == "quantize":
        stats = quantize_mgw(args.input, args.output, args.fraction_bits)
        verify_candidate(args.input, args.output, args.fraction_bits)
        print(json.dumps(stats.to_dict(), sort_keys=True))
        print(f"sha256: {sha256(args.output)}")
    elif args.command == "verify":
        verify_candidate(args.input, args.candidate, args.fraction_bits)
        print("PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PrecisionError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

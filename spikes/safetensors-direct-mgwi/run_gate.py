#!/usr/bin/env python3
"""Run the pinned direct TinyLlama safetensors-to-MGWI identity gate.

This is deliberately a one-candidate gate, not a generic model converter.  It
creates one fresh result directory, invokes the independently
implemented direct converter, compares every output byte with the retained
MGWI, and compares all unpacked weight values with the retained rounded MGW
without writing another 8.8 GB file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import stat
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from safetensors_to_mgwi import Rule, direct_pack  # noqa: E402


SOURCE_BYTES = 2_200_119_864
SOURCE_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
CONFIG_SHA256 = "486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9"
PACKED_BYTES = 2_200_116_192
PACKED_SHA256 = "fa733f5afdec220a91fbaae17ce00bcc20685f25afadd3eec92cfde0af1192c7"
WIDE_BYTES = 8_800_406_496
WIDE_SHA256 = "7ec8e1fd6442c8ad467603460a8cadd3c7d45b3e8fdde94c0cf8b8408fad4b45"
EXPECTED_TENSORS = 201
EXPECTED_ELEMENTS = 1_100_048_384
EXPECTED_F11_TENSORS = 47
EXPECTED_F11_ELEMENTS = 131_164_160
EXPECTED_F12_TENSORS = 154
EXPECTED_F12_ELEMENTS = 968_884_224
DEFAULT_BITS = 11
RULES = [
    Rule("model.layers.*.self_attn.*_proj.weight", 12),
    Rule("model.layers.*.mlp.*_proj.weight", 12),
]

MGW_MAGIC = b"MGW\0"
MGWI_MAGIC = b"MGWI"
FORMAT_VERSION = 1
ENDIAN_TAG = 0x01020304
HEADER = struct.Struct("<4sIIIQQ32s")
CONFIG_BYTES = 64
INDEX = struct.Struct("<64sQQIIII")
COMPARE_BYTES = 8 * 1024 * 1024
COMPARE_ELEMENTS = 1 << 20


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tensor:
    name: str
    elements: int
    offset: int
    ndims: int
    shape0: int
    shape1: int
    tag: int


@dataclass(frozen=True)
class Layout:
    magic: bytes
    size: int
    data_offset: int
    prefix: bytes
    config: bytes
    tensors: tuple[Tensor, ...]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(COMPARE_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def checked_regular(path: Path, expected_size: int, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise GateError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise GateError(f"{label} is not a regular file: {path}")
    if metadata.st_size != expected_size:
        raise GateError(
            f"{label} size mismatch: expected {expected_size}, found {metadata.st_size}"
        )


def read_exact(stream: Any, size: int, label: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise GateError(f"{label}: unexpected end of file")
    return payload


def read_layout(path: Path, expected_magic: bytes, expected_size: int) -> Layout:
    checked_regular(path, expected_size, path.name)
    with path.open("rb") as stream:
        raw_header = read_exact(stream, HEADER.size, f"{path.name} header")
        magic, version, endian, count, index_offset, data_offset, reserved = (
            HEADER.unpack(raw_header)
        )
        if magic != expected_magic:
            raise GateError(f"{path.name}: wrong magic {magic!r}")
        if version != FORMAT_VERSION or endian != ENDIAN_TAG:
            raise GateError(f"{path.name}: unsupported version or endian tag")
        if reserved != bytes(32):
            raise GateError(f"{path.name}: nonzero reserved header bytes")
        if count != EXPECTED_TENSORS:
            raise GateError(
                f"{path.name}: expected {EXPECTED_TENSORS} tensors, found {count}"
            )
        if index_offset != HEADER.size + CONFIG_BYTES:
            raise GateError(f"{path.name}: non-canonical index offset")
        alignment = 8 if expected_magic == MGW_MAGIC else 2
        expected_data_offset = (
            index_offset + count * INDEX.size + alignment - 1
        ) & ~(alignment - 1)
        if data_offset != expected_data_offset:
            raise GateError(f"{path.name}: non-canonical data offset")

        stream.seek(HEADER.size)
        config = read_exact(stream, CONFIG_BYTES, f"{path.name} config")
        stream.seek(index_offset)
        cursor = data_offset
        names: set[str] = set()
        tensors: list[Tensor] = []
        total_elements = 0
        for tensor_index in range(count):
            raw_entry = read_exact(
                stream,
                INDEX.size,
                f"{path.name} index {tensor_index}",
            )
            raw_name, elements, offset, ndims, shape0, shape1, tag = (
                INDEX.unpack(raw_entry)
            )
            name_bytes = raw_name.split(b"\0", 1)[0]
            try:
                name = name_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                raise GateError(f"{path.name}: non-ASCII tensor name") from exc
            if not name or name in names:
                raise GateError(f"{path.name}: empty or duplicate tensor {name!r}")
            names.add(name)
            if ndims == 1:
                expected_elements = shape0
                if shape1 != 0:
                    raise GateError(f"{path.name}: invalid vector shape for {name}")
            elif ndims == 2:
                expected_elements = shape0 * shape1
            else:
                raise GateError(f"{path.name}: invalid rank for {name}")
            if elements != expected_elements or elements == 0:
                raise GateError(f"{path.name}: invalid element count for {name}")
            if offset != cursor:
                raise GateError(f"{path.name}: non-canonical payload offset for {name}")
            if expected_magic == MGW_MAGIC:
                if tag != 0:
                    raise GateError(f"{path.name}: nonzero MGW tag for {name}")
                element_bytes = 8
            else:
                if tag not in (11, 12):
                    raise GateError(f"{path.name}: unexpected MGWI tag F{tag} for {name}")
                element_bytes = 2
            cursor += elements * element_bytes
            total_elements += elements
            tensors.append(
                Tensor(name, elements, offset, ndims, shape0, shape1, tag)
            )
        if cursor != expected_size:
            raise GateError(f"{path.name}: trailing or missing payload bytes")
        if total_elements != EXPECTED_ELEMENTS:
            raise GateError(
                f"{path.name}: expected {EXPECTED_ELEMENTS} elements, "
                f"found {total_elements}"
            )
        stream.seek(0)
        prefix = read_exact(stream, data_offset, f"{path.name} prefix")
    return Layout(
        magic,
        expected_size,
        data_offset,
        prefix,
        config,
        tuple(tensors),
    )


def compare_files(left: Path, right: Path) -> dict[str, Any]:
    checked_regular(left, PACKED_BYTES, "direct MGWI")
    checked_regular(right, PACKED_BYTES, "retained MGWI")
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    compared = 0
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while compared < PACKED_BYTES:
            amount = min(COMPARE_BYTES, PACKED_BYTES - compared)
            left_chunk = read_exact(left_stream, amount, "direct MGWI")
            right_chunk = read_exact(right_stream, amount, "retained MGWI")
            left_digest.update(left_chunk)
            right_digest.update(right_chunk)
            if left_chunk != right_chunk:
                mismatch = next(
                    index
                    for index, (left_byte, right_byte) in enumerate(
                        zip(left_chunk, right_chunk, strict=True)
                    )
                    if left_byte != right_byte
                )
                raise GateError(
                    f"direct and retained MGWI differ at byte {compared + mismatch}"
                )
            compared += amount
    left_sha = left_digest.hexdigest()
    right_sha = right_digest.hexdigest()
    if left_sha != PACKED_SHA256 or right_sha != PACKED_SHA256:
        raise GateError(
            "packed SHA-256 mismatch after full comparison: "
            f"direct={left_sha} retained={right_sha}"
        )
    return {
        "bytes_compared": compared,
        "direct_sha256": left_sha,
        "retained_sha256": right_sha,
        "byte_identical": True,
    }


def compare_unpacked(packed_path: Path, wide_path: Path) -> dict[str, Any]:
    packed = read_layout(packed_path, MGWI_MAGIC, PACKED_BYTES)
    wide = read_layout(wide_path, MGW_MAGIC, WIDE_BYTES)
    if packed.config != wide.config:
        raise GateError("packed and rounded-wide serialized configs differ")
    if len(packed.tensors) != len(wide.tensors):
        raise GateError("packed and rounded-wide tensor counts differ")

    packed_digest = hashlib.sha256(packed.prefix)
    wide_digest = hashlib.sha256(wide.prefix)
    compared_elements = 0
    f_counts = {11: [0, 0], 12: [0, 0]}
    with packed_path.open("rb") as packed_stream, wide_path.open("rb") as wide_stream:
        for packed_tensor, wide_tensor in zip(
            packed.tensors,
            wide.tensors,
            strict=True,
        ):
            if (
                packed_tensor.name != wide_tensor.name
                or packed_tensor.elements != wide_tensor.elements
                or packed_tensor.ndims != wide_tensor.ndims
                or packed_tensor.shape0 != wide_tensor.shape0
                or packed_tensor.shape1 != wide_tensor.shape1
            ):
                raise GateError(
                    f"packed/wide metadata differs at {packed_tensor.name!r}"
                )
            f_counts[packed_tensor.tag][0] += 1
            f_counts[packed_tensor.tag][1] += packed_tensor.elements
            packed_stream.seek(packed_tensor.offset)
            wide_stream.seek(wide_tensor.offset)
            tensor_cursor = 0
            while tensor_cursor < packed_tensor.elements:
                count = min(
                    COMPARE_ELEMENTS,
                    packed_tensor.elements - tensor_cursor,
                )
                packed_raw = read_exact(
                    packed_stream,
                    count * 2,
                    f"packed payload {packed_tensor.name}",
                )
                wide_raw = read_exact(
                    wide_stream,
                    count * 8,
                    f"wide payload {wide_tensor.name}",
                )
                packed_digest.update(packed_raw)
                wide_digest.update(wide_raw)
                codes = np.frombuffer(packed_raw, dtype="<i2")
                actual_wide = np.frombuffer(wide_raw, dtype="<i8")
                expected_wide = codes.astype(np.int64) * (
                    1 << (48 - packed_tensor.tag)
                )
                unequal = np.flatnonzero(expected_wide != actual_wide)
                if unequal.size:
                    element = int(unequal[0])
                    raise GateError(
                        f"unpacked mismatch in {packed_tensor.name} at element "
                        f"{tensor_cursor + element}: expected "
                        f"{int(expected_wide[element])}, found "
                        f"{int(actual_wide[element])}"
                    )
                tensor_cursor += count
                compared_elements += count

    packed_sha = packed_digest.hexdigest()
    wide_sha = wide_digest.hexdigest()
    if packed_sha != PACKED_SHA256:
        raise GateError(f"packed SHA-256 mismatch: {packed_sha}")
    if wide_sha != WIDE_SHA256:
        raise GateError(f"rounded-wide SHA-256 mismatch: {wide_sha}")
    if f_counts[11] != [EXPECTED_F11_TENSORS, EXPECTED_F11_ELEMENTS]:
        raise GateError(f"unexpected F11 distribution: {f_counts[11]}")
    if f_counts[12] != [EXPECTED_F12_TENSORS, EXPECTED_F12_ELEMENTS]:
        raise GateError(f"unexpected F12 distribution: {f_counts[12]}")
    if compared_elements != EXPECTED_ELEMENTS:
        raise GateError(
            f"expected {EXPECTED_ELEMENTS} compared elements, found {compared_elements}"
        )
    return {
        "elements_compared": compared_elements,
        "packed_sha256": packed_sha,
        "rounded_wide_sha256": wide_sha,
        "unpacked_byte_identical": True,
        "F11": {
            "tensors": f_counts[11][0],
            "elements": f_counts[11][1],
        },
        "F12": {
            "tensors": f_counts[12][0],
            "elements": f_counts[12][1],
        },
    }


def create_result_directory(path: Path) -> Path:
    parent = path.parent.resolve(strict=True)
    parent_mode = parent.stat().st_mode
    if parent_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise GateError(f"result parent must not be group/world writable: {parent}")
    result = parent / path.name
    try:
        result.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise GateError(f"refusing to reuse result directory: {result}") from exc
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(raw):
            count = os.write(fd, raw[written:])
            if count <= 0:
                raise GateError("short write while recording gate result")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)


def run(args: argparse.Namespace) -> dict[str, Any]:
    converter_path = ROOT / "tools" / "safetensors_to_mgwi.py"
    gate_path = Path(__file__).resolve()
    tool_hashes = {
        "converter_sha256": sha256_path(converter_path),
        "gate_sha256": sha256_path(gate_path),
    }
    result_dir = create_result_directory(args.output_dir)
    candidate = result_dir / "tinyllama-direct-f11-f12.mgwi"
    model_source = args.model_dir / "model.safetensors"
    model_config = args.model_dir / "config.json"

    checked_regular(model_source, SOURCE_BYTES, "source safetensors")
    checked_regular(args.retained_mgwi, PACKED_BYTES, "retained MGWI")
    checked_regular(args.rounded_mgw, WIDE_BYTES, "rounded-wide MGW")
    if sha256_path(model_config) != CONFIG_SHA256:
        raise GateError("config.json SHA-256 mismatch")
    free_bytes = os.statvfs(result_dir).f_bavail * os.statvfs(result_dir).f_frsize
    if free_bytes < PACKED_BYTES + 1_073_741_824:
        raise GateError(
            f"insufficient free space: need at least {PACKED_BYTES + 1_073_741_824}, "
            f"found {free_bytes}"
        )

    print("DIRECT_MGWI_GATE phase=convert", flush=True)
    convert_started = time.monotonic()
    conversion = direct_pack(
        args.model_dir,
        candidate,
        DEFAULT_BITS,
        RULES,
        expected_source_sha256=SOURCE_SHA256,
        expected_config_sha256=CONFIG_SHA256,
        expected_output_sha256=PACKED_SHA256,
    )
    convert_seconds = time.monotonic() - convert_started

    if conversion["output_bytes"] != PACKED_BYTES:
        raise GateError("direct converter reported an unexpected output size")
    if conversion["num_tensors"] != EXPECTED_TENSORS:
        raise GateError("direct converter reported an unexpected tensor count")
    if conversion["num_elements"] != EXPECTED_ELEMENTS:
        raise GateError("direct converter reported an unexpected element count")

    print("DIRECT_MGWI_GATE phase=full-byte-compare", flush=True)
    bytes_started = time.monotonic()
    byte_comparison = compare_files(candidate, args.retained_mgwi)
    byte_compare_seconds = time.monotonic() - bytes_started

    print("DIRECT_MGWI_GATE phase=unpacked-element-compare", flush=True)
    unpacked_started = time.monotonic()
    unpacked_comparison = compare_unpacked(candidate, args.rounded_mgw)
    unpacked_compare_seconds = time.monotonic() - unpacked_started
    if sha256_path(converter_path) != tool_hashes["converter_sha256"]:
        raise GateError("converter source changed during the gate")
    if sha256_path(gate_path) != tool_hashes["gate_sha256"]:
        raise GateError("gate source changed during the gate")

    result = {
        "schema": "direct_safetensors_mgwi_gate_v1",
        "status": "PASS",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host_class": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "plan": {
            "default_fraction_bits": DEFAULT_BITS,
            "rules": {rule.pattern: rule.bits for rule in RULES},
            "physical_storage": "little-endian signed int16",
        },
        "inputs": {
            "source_safetensors": {
                "bytes": SOURCE_BYTES,
                "sha256": SOURCE_SHA256,
            },
            "config_json": {
                "bytes": model_config.stat().st_size,
                "sha256": CONFIG_SHA256,
            },
            "retained_mgwi": {
                "bytes": PACKED_BYTES,
                "sha256": PACKED_SHA256,
            },
            "rounded_wide_mgw": {
                "bytes": WIDE_BYTES,
                "sha256": WIDE_SHA256,
            },
        },
        "tools": tool_hashes,
        "conversion": conversion,
        "full_file_comparison": byte_comparison,
        "unpacked_comparison": unpacked_comparison,
        "timing_seconds": {
            "conversion": convert_seconds,
            "full_byte_comparison": byte_compare_seconds,
            "unpacked_element_comparison": unpacked_compare_seconds,
            "total": convert_seconds + byte_compare_seconds + unpacked_compare_seconds,
        },
    }
    write_result(result_dir / "gate-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--retained-mgwi", type=Path, required=True)
    parser.add_argument("--rounded-mgw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    timing = result["timing_seconds"]
    print(
        "DIRECT_MGWI_GATE PASS "
        f"bytes={result['full_file_comparison']['bytes_compared']} "
        f"elements={result['unpacked_comparison']['elements_compared']} "
        f"sha256={result['conversion']['output_sha256']} "
        f"seconds={timing['total']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, OSError, ValueError) as exc:
        print(f"DIRECT_MGWI_GATE FAIL: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

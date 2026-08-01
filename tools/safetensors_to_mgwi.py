#!/usr/bin/env python3
"""Convert one Llama safetensors checkpoint directly to tagged-int16 MGWI.

The conversion deliberately reproduces the existing reference pipeline:
source float bits -> Q16.48 -> coarser F-grid rounding -> signed-int16 code.
Only bounded chunks are expanded to Q16.48 in memory; no wide model file is
created.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import mmap
import os
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FP_FRACTION_BITS = 48
CHUNK_ELEMENTS = 1 << 20
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_TENSORS = 512
INT64_MAX = (1 << 63) - 1
MGWI_MAGIC = b"MGWI"
FORMAT_VERSION = 1
ENDIAN_TAG = 0x01020304
HEADER = struct.Struct("<4sIIIQQ32s")
CONFIG_SIZE = 64
INDEX = struct.Struct("<64sQQIIII")
CONFIG = struct.Struct("<16i")
SAFETENSORS_LENGTH = struct.Struct("<Q")
SUPPORTED_DTYPES = {"BF16", "F16"}


class FormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class Rule:
    pattern: str
    bits: int


@dataclass(frozen=True)
class OutputTensor:
    name: str
    num_elements: int
    ndims: int
    shape0: int
    shape1: int


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def parse_set_rule(text: str) -> Rule:
    pattern, separator, raw_bits = text.rpartition("=")
    if not separator or not pattern:
        raise argparse.ArgumentTypeError("expected GLOB=BITS")
    try:
        bits = int(raw_bits)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("BITS must be an integer") from exc
    if not 0 <= bits <= FP_FRACTION_BITS:
        raise argparse.ArgumentTypeError("BITS must be in [0, 48]")
    return Rule(pattern, bits)


def choose_plan(
    tensors: tuple[OutputTensor, ...],
    default_bits: int,
    rules: list[Rule],
) -> list[int]:
    matches = [0] * len(rules)
    plan: list[int] = []
    for tensor in tensors:
        choice = default_bits
        for rule_index, rule in enumerate(rules):
            if fnmatch.fnmatchcase(tensor.name, rule.pattern):
                choice = rule.bits
                matches[rule_index] += 1
        plan.append(choice)
    for rule, count in zip(rules, matches, strict=True):
        if count == 0:
            raise FormatError(f"rule matched no tensors: {rule.pattern}")
    return plan


@dataclass(frozen=True)
class LlamaConfig:
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    rope_theta: int


@dataclass(frozen=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    num_elements: int
    absolute_offset: int


@dataclass(frozen=True)
class SourceModel:
    model_path: Path
    size: int
    tensors: dict[str, SourceTensor]


def checked_sha256(text: str) -> str:
    value = text.lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("expected a 64-digit SHA-256")
    return value


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormatError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise FormatError(f"{label}: non-JSON numeric constant {value!r}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"{label}: invalid JSON: {exc}") from exc


def integral_field(
    obj: dict[str, Any],
    key: str,
    default: int | None = None,
) -> int:
    if key not in obj:
        if default is None:
            raise FormatError(f"config.json: missing {key!r}")
        return default
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormatError(f"config.json: {key!r} must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise FormatError(f"config.json: {key!r} must be an exact integer")
    integer = int(value)
    if integer < -(1 << 31) or integer > (1 << 31) - 1:
        raise FormatError(f"config.json: {key!r} is outside int32")
    return integer


def read_config(model_dir: Path) -> tuple[LlamaConfig, bytes, str]:
    path = model_dir / "config.json"
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise FormatError(f"missing {path}") from exc
    if size <= 0 or size > 1024 * 1024:
        raise FormatError(f"{path}: invalid size {size}")
    raw = path.read_bytes()
    parsed = parse_json(raw, str(path))
    if not isinstance(parsed, dict):
        raise FormatError(f"{path}: top level must be an object")

    hidden = integral_field(parsed, "hidden_size")
    heads = integral_field(parsed, "num_attention_heads")
    kv_heads = integral_field(parsed, "num_key_value_heads", heads)
    layers = integral_field(parsed, "num_hidden_layers")
    intermediate = integral_field(parsed, "intermediate_size")
    vocab = integral_field(parsed, "vocab_size")
    max_seq = integral_field(parsed, "max_position_embeddings", 2048)
    rope_theta = integral_field(parsed, "rope_theta", 10000)
    if min(hidden, heads, kv_heads, layers, intermediate, vocab, max_seq) <= 0:
        raise FormatError("config.json: model dimensions must be positive")
    if hidden % heads != 0:
        raise FormatError("config.json: hidden_size is not divisible by head count")
    if heads % kv_heads != 0:
        raise FormatError("config.json: attention heads are not divisible by KV heads")
    config = LlamaConfig(
        hidden,
        heads,
        kv_heads,
        hidden // heads,
        layers,
        intermediate,
        vocab,
        max_seq,
        rope_theta,
    )
    return config, raw, hashlib.sha256(raw).hexdigest()


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FormatError(f"{label}: expected a positive integer")
    return value


def read_exact_at(fd: int, size: int, offset: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    cursor = offset
    while remaining:
        chunk = os.pread(fd, remaining, cursor)
        if not chunk:
            raise FormatError(f"{label}: unexpected end of file")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def sha256_fd(fd: int, size: int, label: str) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < size:
        chunk = os.pread(fd, min(8 * 1024 * 1024, size - cursor), cursor)
        if not chunk:
            raise FormatError(f"{label}: truncated while hashing")
        digest.update(chunk)
        cursor += len(chunk)
    current = os.fstat(fd)
    if current.st_size != size:
        raise FormatError(f"{label}: size changed while hashing")
    return digest.hexdigest()


def read_safetensors_fd(path: Path, fd: int) -> SourceModel:
    file_stat = os.fstat(fd)
    if not stat.S_ISREG(file_stat.st_mode):
        raise FormatError(f"{path}: source is not a regular file")
    size = file_stat.st_size
    if size < SAFETENSORS_LENGTH.size:
        raise FormatError(f"{path}: truncated length prefix")
    raw_length = read_exact_at(
        fd,
        SAFETENSORS_LENGTH.size,
        0,
        str(path),
    )
    (header_size,) = SAFETENSORS_LENGTH.unpack(raw_length)
    if header_size == 0 or header_size > MAX_JSON_BYTES:
        raise FormatError(f"{path}: invalid header size {header_size}")
    if header_size > size - SAFETENSORS_LENGTH.size:
        raise FormatError(f"{path}: header exceeds file size")
    raw_header = read_exact_at(
        fd,
        header_size,
        SAFETENSORS_LENGTH.size,
        str(path),
    )
    parsed = parse_json(raw_header, str(path))
    if not isinstance(parsed, dict):
        raise FormatError(f"{path}: top level must be an object")

    data_start = SAFETENSORS_LENGTH.size + header_size
    data_size = size - data_start
    tensors: dict[str, SourceTensor] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, descriptor in parsed.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                raise FormatError(f"{path}: __metadata__ must be an object")
            if any(not isinstance(value, str) for value in descriptor.values()):
                raise FormatError(f"{path}: __metadata__ values must be strings")
            continue
        try:
            encoded_name = name.encode("ascii")
        except (AttributeError, UnicodeEncodeError) as exc:
            raise FormatError(f"{path}: tensor names must be ASCII strings") from exc
        if not encoded_name or len(encoded_name) > 63:
            raise FormatError(f"{path}: invalid MGWI tensor name {name!r}")
        if not isinstance(descriptor, dict):
            raise FormatError(f"{path}: tensor {name!r} descriptor is not an object")
        descriptor_keys = set(descriptor)
        if descriptor_keys != {"dtype", "shape", "data_offsets"}:
            raise FormatError(
                f"{path}: tensor {name!r} has unexpected descriptor fields"
            )
        dtype = descriptor.get("dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise FormatError(f"{path}: tensor {name!r} has unsupported dtype {dtype!r}")
        raw_shape = descriptor.get("shape")
        if not isinstance(raw_shape, list) or not 1 <= len(raw_shape) <= 2:
            raise FormatError(f"{path}: tensor {name!r} must be one- or two-dimensional")
        shape = tuple(
            positive_integer(dimension, f"{path}: {name!r} shape")
            for dimension in raw_shape
        )
        raw_offsets = descriptor.get("data_offsets")
        if (
            not isinstance(raw_offsets, list)
            or len(raw_offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_offsets)
        ):
            raise FormatError(f"{path}: tensor {name!r} has invalid data_offsets")
        start, end = raw_offsets
        elements = math.prod(shape)
        expected_bytes = elements * 2
        if start < 0 or end < start or end - start != expected_bytes:
            raise FormatError(f"{path}: tensor {name!r} payload size mismatch")
        if end > data_size:
            raise FormatError(f"{path}: tensor {name!r} payload exceeds file")
        tensors[name] = SourceTensor(
            name,
            dtype,
            shape,
            elements,
            data_start + start,
        )
        intervals.append((start, end, name))

    if len(tensors) > MAX_OUTPUT_TENSORS:
        raise FormatError(
            f"{path}: tensor count {len(tensors)} exceeds {MAX_OUTPUT_TENSORS}"
        )

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise FormatError(
                f"{path}: non-canonical gap or overlap before tensor {name!r}"
            )
        cursor = end
    if cursor != data_size:
        raise FormatError(f"{path}: trailing or unclaimed tensor data")
    if not tensors:
        raise FormatError(f"{path}: no tensors")
    return SourceModel(path, size, tensors)


def canonical_tensors(
    config: LlamaConfig,
    source: SourceModel,
) -> tuple[tuple[OutputTensor, ...], tuple[SourceTensor, ...], int]:
    kv_dim = config.num_kv_heads * config.head_dim
    lm_head_tied = int("lm_head.weight" not in source.tensors)
    expected_tensor_count = (2 if lm_head_tied else 3) + 9 * config.num_layers
    if expected_tensor_count > MAX_OUTPUT_TENSORS:
        raise FormatError(
            f"derived tensor count {expected_tensor_count} exceeds "
            f"{MAX_OUTPUT_TENSORS}"
        )
    specifications: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (config.vocab_size, config.hidden_dim)),
        ("model.norm.weight", (config.hidden_dim,)),
    ]
    if not lm_head_tied:
        specifications.append(
            ("lm_head.weight", (config.vocab_size, config.hidden_dim))
        )
    for layer in range(config.num_layers):
        prefix = f"model.layers.{layer}."
        specifications.extend(
            [
                (prefix + "input_layernorm.weight", (config.hidden_dim,)),
                (prefix + "self_attn.q_proj.weight", (config.hidden_dim, config.hidden_dim)),
                (prefix + "self_attn.k_proj.weight", (kv_dim, config.hidden_dim)),
                (prefix + "self_attn.v_proj.weight", (kv_dim, config.hidden_dim)),
                (prefix + "self_attn.o_proj.weight", (config.hidden_dim, config.hidden_dim)),
                (prefix + "post_attention_layernorm.weight", (config.hidden_dim,)),
                (prefix + "mlp.gate_proj.weight", (config.intermediate_dim, config.hidden_dim)),
                (prefix + "mlp.up_proj.weight", (config.intermediate_dim, config.hidden_dim)),
                (prefix + "mlp.down_proj.weight", (config.hidden_dim, config.intermediate_dim)),
            ]
        )

    output_tensors: list[OutputTensor] = []
    source_tensors: list[SourceTensor] = []
    for name, shape in specifications:
        tensor = source.tensors.get(name)
        if tensor is None:
            raise FormatError(f"source checkpoint is missing {name!r}")
        if tensor.shape != shape:
            raise FormatError(
                f"{name}: expected shape {shape}, found {tensor.shape}"
            )
        output_tensors.append(
            OutputTensor(
                name,
                tensor.num_elements,
                len(shape),
                shape[0],
                shape[1] if len(shape) == 2 else 0,
            )
        )
        source_tensors.append(tensor)
    return tuple(output_tensors), tuple(source_tensors), lm_head_tied


def config_bytes(config: LlamaConfig, lm_head_tied: int) -> bytes:
    return CONFIG.pack(
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.num_layers,
        config.intermediate_dim,
        config.vocab_size,
        config.max_seq_len,
        config.rope_theta,
        lm_head_tied,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def source_chunk_to_q1648(
    raw: np.ndarray,
    dtype: str,
    tensor_name: str,
) -> np.ndarray:
    bits = np.asarray(raw, dtype=np.uint16)
    sign = (bits >> 15) != 0
    result = np.zeros(bits.shape, dtype=np.int64)

    if dtype == "BF16":
        exponent = ((bits >> 7) & 0xFF).astype(np.int16)
        mantissa = (bits & 0x7F).astype(np.int64)
        special = exponent == 0xFF
        if np.any(special):
            raise FormatError(f"{tensor_name}: BF16 contains non-finite value")
        normal = (exponent != 0) & ~special
        shifts = exponent.astype(np.int16) - 86
        significand = mantissa + 128
        for raw_shift in np.unique(shifts[normal]):
            shift = int(raw_shift)
            mask = normal & (shifts == shift)
            if shift >= 64:
                raise FormatError(
                    f"{tensor_name}: BF16 value overflows Q16.48 int64"
                )
            if shift >= 0:
                values = significand[mask]
                if values.size and int(values.max()) > (INT64_MAX >> shift):
                    raise FormatError(
                        f"{tensor_name}: BF16 value overflows Q16.48 int64"
                    )
                result[mask] = values << shift
            elif -64 < shift < 0:
                result[mask] = significand[mask] >> (-shift)
            else:
                result[mask] = 0
    elif dtype == "F16":
        exponent = ((bits >> 10) & 0x1F).astype(np.int16)
        mantissa = (bits & 0x3FF).astype(np.int64)
        special = exponent == 0x1F
        if np.any(special):
            raise FormatError(f"{tensor_name}: F16 contains non-finite value")
        subnormal = (exponent == 0) & (mantissa != 0)
        result[subnormal] = mantissa[subnormal] << 24
        normal = (exponent != 0) & ~special
        shifts = exponent.astype(np.int16) + 23
        significand = mantissa + 1024
        for raw_shift in np.unique(shifts[normal]):
            shift = int(raw_shift)
            mask = normal & (shifts == shift)
            values = significand[mask]
            if values.size and int(values.max()) > (INT64_MAX >> shift):
                raise FormatError(
                    f"{tensor_name}: F16 value overflows Q16.48 int64"
                )
            result[mask] = values << shift
    else:
        raise FormatError(f"{tensor_name}: unsupported dtype {dtype}")

    negative = sign & (result != 0)
    result[negative] = -result[negative]
    return result


def q1648_to_int16_codes(
    values: np.ndarray,
    fraction_bits: int,
    tensor_name: str,
) -> np.ndarray:
    if not 0 <= fraction_bits <= FP_FRACTION_BITS:
        raise FormatError(f"{tensor_name}: invalid F{fraction_bits} tag")
    if np.any(values == np.iinfo(np.int64).min):
        raise FormatError(f"{tensor_name}: INT64_MIN cannot be rounded safely")
    negative = values < 0
    magnitude = np.where(negative, -values, values)
    dropped = FP_FRACTION_BITS - fraction_bits
    if dropped == 0:
        rounded = magnitude
    else:
        base = magnitude >> dropped
        remainder = magnitude - (base << dropped)
        rounded = base + (remainder >= (1 << (dropped - 1)))
    positive_overflow = ~negative & (rounded > 32767)
    negative_overflow = negative & (rounded > 32768)
    if np.any(positive_overflow | negative_overflow):
        signed = np.where(negative, -rounded, rounded)
        raise FormatError(
            f"{tensor_name}: F{fraction_bits} code range "
            f"{int(signed.min())}..{int(signed.max())} does not fit int16"
        )
    signed = np.where(negative, -rounded, rounded)
    return signed.astype("<i2")


def packed_layout(
    tensors: tuple[OutputTensor, ...],
    plan: list[int],
) -> tuple[bytes, int, tuple[int, ...]]:
    """Serialize the MGWI index without using the reference packer's code."""
    data_offset = align(
        HEADER.size + CONFIG_SIZE + len(tensors) * INDEX.size,
        2,
    )
    cursor = data_offset
    entries: list[bytes] = []
    offsets: list[int] = []
    for tensor, fraction_bits in zip(tensors, plan, strict=True):
        try:
            raw_name = tensor.name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise FormatError(f"non-ASCII tensor name {tensor.name!r}") from exc
        if not raw_name or len(raw_name) > 63:
            raise FormatError(f"invalid MGWI tensor name {tensor.name!r}")
        offsets.append(cursor)
        entries.append(
            INDEX.pack(
                raw_name,
                tensor.num_elements,
                cursor,
                tensor.ndims,
                tensor.shape0,
                tensor.shape1,
                fraction_bits,
            )
        )
        cursor += tensor.num_elements * 2
    return b"".join(entries), cursor, tuple(offsets)


def pwrite_all(fd: int, payload: bytes, offset: int, label: str) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:], offset + written)
        if count <= 0:
            raise FormatError(f"{label}: short pwrite")
        written += count


def write_prefix(
    fd: int,
    num_tensors: int,
    data_offset: int,
    raw_config: bytes,
    raw_index: bytes,
) -> None:
    if len(raw_config) != CONFIG_SIZE:
        raise FormatError("internal error: invalid serialized config size")
    header = HEADER.pack(
        MGWI_MAGIC,
        FORMAT_VERSION,
        ENDIAN_TAG,
        num_tensors,
        HEADER.size + CONFIG_SIZE,
        data_offset,
        bytes(32),
    )
    pwrite_all(fd, header, 0, "MGWI header")
    pwrite_all(fd, raw_config, HEADER.size, "MGWI config")
    pwrite_all(fd, raw_index, HEADER.size + CONFIG_SIZE, "MGWI index")


def secure_destination(destination: Path) -> tuple[Path, Path]:
    """Resolve a private output directory and refuse any existing final path."""
    if destination.name in ("", ".", ".."):
        raise FormatError(f"invalid destination {destination}")
    try:
        parent = destination.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FormatError(f"destination directory does not exist: {destination.parent}") from exc
    parent_stat = parent.stat()
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise FormatError(f"destination parent is not a directory: {parent}")
    if parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise FormatError(
            f"destination directory must not be group/world writable: {parent}"
        )
    final_path = parent / destination.name
    try:
        os.lstat(final_path)
    except FileNotFoundError:
        pass
    else:
        raise FormatError(f"refusing to overwrite {final_path}")
    return parent, final_path


def convert_tensor(
    source_map: mmap.mmap,
    output_map: mmap.mmap,
    source_tensor: SourceTensor,
    output_offset: int,
    fraction_bits: int,
) -> tuple[int, int]:
    source_values: np.ndarray | None = None
    output_values: np.ndarray | None = None
    try:
        source_values = np.ndarray(
            (source_tensor.num_elements,),
            dtype="<u2",
            buffer=source_map,
            offset=source_tensor.absolute_offset,
        )
        output_values = np.ndarray(
            (source_tensor.num_elements,),
            dtype="<i2",
            buffer=output_map,
            offset=output_offset,
        )
        minimum: int | None = None
        maximum: int | None = None
        for start in range(0, source_tensor.num_elements, CHUNK_ELEMENTS):
            end = min(start + CHUNK_ELEMENTS, source_tensor.num_elements)
            q1648 = source_chunk_to_q1648(
                source_values[start:end],
                source_tensor.dtype,
                source_tensor.name,
            )
            codes = q1648_to_int16_codes(
                q1648,
                fraction_bits,
                source_tensor.name,
            )
            output_values[start:end] = codes
            chunk_minimum = int(codes.min())
            chunk_maximum = int(codes.max())
            minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
            maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)
            del codes
            del q1648
        if minimum is None or maximum is None:
            raise FormatError(f"{source_tensor.name}: empty tensor")
        return minimum, maximum
    finally:
        output_values = None
        source_values = None


def fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def direct_pack(
    model_dir: Path,
    destination: Path,
    default_bits: int,
    rules: list[Rule],
    expected_source_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_output_sha256: str | None = None,
) -> dict[str, Any]:
    parent, final_path = secure_destination(destination)
    config, _raw_source_config, config_sha256 = read_config(model_dir)
    if expected_config_sha256 is not None and config_sha256 != expected_config_sha256:
        raise FormatError(
            "config SHA-256 mismatch: "
            f"expected {expected_config_sha256}, found {config_sha256}"
        )

    source_path = model_dir / "model.safetensors"
    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_fd = -1
    output_fd = -1
    source_map: mmap.mmap | None = None
    output_map: mmap.mmap | None = None
    temporary_path: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    linked_final = False
    try:
        source_fd = os.open(source_path, source_flags)
        source_before = os.fstat(source_fd)
        source = read_safetensors_fd(source_path, source_fd)
        source_sha256 = sha256_fd(source_fd, source.size, str(source_path))
        if (
            expected_source_sha256 is not None
            and source_sha256 != expected_source_sha256
        ):
            raise FormatError(
                "source SHA-256 mismatch: "
                f"expected {expected_source_sha256}, found {source_sha256}"
            )

        tensors, source_tensors, tied = canonical_tensors(config, source)
        plan = choose_plan(tensors, default_bits, rules)
        raw_index, output_size, output_offsets = packed_layout(tensors, plan)
        data_offset = align(
            HEADER.size + CONFIG_SIZE + len(tensors) * INDEX.size,
            2,
        )
        raw_config = config_bytes(config, tied)

        output_fd, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{final_path.name}.tmp-",
            dir=parent,
        )
        temporary_path = Path(raw_temporary_path)
        temporary_stat = os.fstat(output_fd)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.ftruncate(output_fd, output_size)
        write_prefix(
            output_fd,
            len(tensors),
            data_offset,
            raw_config,
            raw_index,
        )

        source_map = mmap.mmap(source_fd, 0, access=mmap.ACCESS_READ)
        output_map = mmap.mmap(output_fd, 0, access=mmap.ACCESS_WRITE)
        ranges: dict[int, list[int]] = {}
        for source_tensor, output_offset, selected_bits in zip(
            source_tensors,
            output_offsets,
            plan,
            strict=True,
        ):
            minimum, maximum = convert_tensor(
                source_map,
                output_map,
                source_tensor,
                output_offset,
                selected_bits,
            )
            if selected_bits not in ranges:
                ranges[selected_bits] = [
                    1,
                    source_tensor.num_elements,
                    minimum,
                    maximum,
                ]
            else:
                aggregate = ranges[selected_bits]
                aggregate[0] += 1
                aggregate[1] += source_tensor.num_elements
                aggregate[2] = min(aggregate[2], minimum)
                aggregate[3] = max(aggregate[3], maximum)

        output_map.flush()
        output_map.close()
        output_map = None
        source_map.close()
        source_map = None
        os.fsync(output_fd)

        source_after = os.fstat(source_fd)
        if not same_file_identity(source_before, source_after):
            raise FormatError("source file identity or metadata changed during conversion")
        source_after_sha256 = sha256_fd(source_fd, source.size, str(source_path))
        if source_after_sha256 != source_sha256:
            raise FormatError("source file content changed during conversion")

        output_sha256 = sha256_fd(output_fd, output_size, str(temporary_path))
        if (
            expected_output_sha256 is not None
            and output_sha256 != expected_output_sha256
        ):
            raise FormatError(
                "output SHA-256 mismatch: "
                f"expected {expected_output_sha256}, found {output_sha256}"
            )

        os.fchmod(output_fd, 0o644)
        os.link(temporary_path, final_path, follow_symlinks=False)
        linked_final = True
        final_stat = os.lstat(final_path)
        current_temporary_stat = os.fstat(output_fd)
        if (
            final_stat.st_dev != current_temporary_stat.st_dev
            or final_stat.st_ino != current_temporary_stat.st_ino
        ):
            raise FormatError("published output inode does not match temporary output")
        os.unlink(temporary_path)
        temporary_path = None
        # The final path now names the fully validated inode.  Leave it in
        # place even if the directory fsync itself reports an I/O error.
        linked_final = False
        fsync_directory(parent)

        return {
            "source_bytes": source.size,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "output_bytes": output_size,
            "output_sha256": output_sha256,
            "num_tensors": len(tensors),
            "num_elements": sum(tensor.num_elements for tensor in tensors),
            "lm_head_tied": tied,
            "ranges": ranges,
        }
    finally:
        if output_map is not None:
            output_map.close()
        if source_map is not None:
            source_map.close()
        if output_fd >= 0:
            os.close(output_fd)
        if source_fd >= 0:
            os.close(source_fd)
        if temporary_path is not None and temporary_identity is not None:
            try:
                current = os.lstat(temporary_path)
                if (current.st_dev, current.st_ino) == temporary_identity:
                    os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        if linked_final:
            # A failure after link but before unlinking the hidden name must not
            # leave a candidate at the public path.  Only remove our own inode.
            try:
                current = os.lstat(final_path)
                if temporary_identity == (current.st_dev, current.st_ino):
                    os.unlink(final_path)
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-bits", type=int, required=True)
    parser.add_argument(
        "--set",
        dest="rules",
        action="append",
        type=parse_set_rule,
        default=[],
        metavar="GLOB=BITS",
    )
    parser.add_argument(
        "--expect-source-sha256",
        type=checked_sha256,
        required=True,
    )
    parser.add_argument(
        "--expect-config-sha256",
        type=checked_sha256,
        required=True,
    )
    parser.add_argument(
        "--expect-output-sha256",
        type=checked_sha256,
        required=True,
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not 0 <= args.default_bits <= FP_FRACTION_BITS:
        raise FormatError("--default-bits must be in [0, 48]")
    result = direct_pack(
        args.model_dir,
        args.destination,
        args.default_bits,
        list(args.rules),
        args.expect_source_sha256,
        args.expect_config_sha256,
        args.expect_output_sha256,
    )
    print(f"source_bytes={result['source_bytes']}")
    print(f"source_sha256={result['source_sha256']}")
    print(f"config_sha256={result['config_sha256']}")
    print(f"output_bytes={result['output_bytes']}")
    print(f"output_sha256={result['output_sha256']}")
    print(f"tensors={result['num_tensors']}")
    print(f"elements={result['num_elements']}")
    for bits, values in sorted(result["ranges"].items()):
        tensor_count, element_count, minimum, maximum = values
        print(
            f"F{bits} tensors={tensor_count} elements={element_count} "
            f"code_min={minimum} code_max={maximum}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormatError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

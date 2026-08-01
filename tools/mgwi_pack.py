#!/usr/bin/env python3
"""Pack canonical Q16.48 MGW tensors as tagged signed-int16 payloads.

This is an experimental storage-format spike.  It does not change the
inference arithmetic: unpacking must recreate the input rounded MGW
byte-for-byte.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FP_FRACTION_BITS = 48
MGW_MAGIC = b"MGW\0"
MGWI_MAGIC = b"MGWI"
FORMAT_VERSION = 1
ENDIAN_TAG = 0x01020304
RAW_INT64_TAG = 0xFFFFFFFF
CHUNK_ELEMENTS = 1 << 20

HEADER = struct.Struct("<4sIIIQQ32s")
CONFIG_SIZE = 64
INDEX = struct.Struct("<64sQQIIII")


class FormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tensor:
    raw_name: bytes
    name: str
    num_elements: int
    data_offset: int
    ndims: int
    shape0: int
    shape1: int
    tag: int


@dataclass(frozen=True)
class Layout:
    path: Path
    size: int
    magic: bytes
    num_tensors: int
    config: bytes
    tensors: tuple[Tensor, ...]


@dataclass(frozen=True)
class Rule:
    pattern: str
    bits: int | None


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def read_layout(path: Path, expected_magic: bytes) -> Layout:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_header = stream.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise FormatError(f"{path}: truncated header")
        (
            magic,
            version,
            endian_tag,
            num_tensors,
            index_offset,
            data_offset,
            _reserved,
        ) = HEADER.unpack(raw_header)
        if magic != expected_magic:
            raise FormatError(f"{path}: unexpected magic {magic!r}")
        if version != FORMAT_VERSION or endian_tag != ENDIAN_TAG:
            raise FormatError(f"{path}: unsupported version or byte order")
        if num_tensors == 0 or num_tensors > 1_000_000:
            raise FormatError(f"{path}: invalid tensor count")
        if index_offset != HEADER.size + CONFIG_SIZE:
            raise FormatError(f"{path}: non-canonical index offset")
        expected_data_offset = align(
            index_offset + num_tensors * INDEX.size,
            8 if expected_magic == MGW_MAGIC else 2,
        )
        if data_offset != expected_data_offset:
            raise FormatError(f"{path}: non-canonical data offset")

        stream.seek(HEADER.size)
        config = stream.read(CONFIG_SIZE)
        if len(config) != CONFIG_SIZE:
            raise FormatError(f"{path}: truncated config")
        stream.seek(index_offset)
        tensors: list[Tensor] = []
        names: set[str] = set()
        expected_offset = data_offset
        for tensor_index in range(num_tensors):
            raw_entry = stream.read(INDEX.size)
            if len(raw_entry) != INDEX.size:
                raise FormatError(
                    f"{path}: truncated tensor index at {tensor_index}"
                )
            (
                raw_name,
                num_elements,
                tensor_offset,
                ndims,
                shape0,
                shape1,
                tag,
            ) = INDEX.unpack(raw_entry)
            name_bytes = raw_name.split(b"\0", 1)[0]
            try:
                name = name_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                raise FormatError(f"{path}: non-ASCII tensor name") from exc
            if not name or name in names:
                raise FormatError(f"{path}: invalid/duplicate tensor {name!r}")
            names.add(name)
            expected_elements = (
                shape0 if ndims == 1 else shape0 * shape1
            )
            if ndims not in (1, 2) or num_elements != expected_elements:
                raise FormatError(f"{path}: invalid shape for {name}")
            if tensor_offset != expected_offset:
                raise FormatError(
                    f"{path}: non-canonical payload offset for {name}"
                )
            if expected_magic == MGW_MAGIC:
                if tag != 0:
                    raise FormatError(f"{path}: nonzero MGW tag for {name}")
                element_bytes = 8
            else:
                if tag != RAW_INT64_TAG and not 0 <= tag <= FP_FRACTION_BITS:
                    raise FormatError(f"{path}: invalid MGWI tag for {name}")
                element_bytes = 8 if tag == RAW_INT64_TAG else 2
            payload_bytes = num_elements * element_bytes
            if tensor_offset > size or payload_bytes > size - tensor_offset:
                raise FormatError(f"{path}: payload outside file for {name}")
            expected_offset += payload_bytes
            tensors.append(
                Tensor(
                    raw_name,
                    name,
                    num_elements,
                    tensor_offset,
                    ndims,
                    shape0,
                    shape1,
                    tag,
                )
            )
        if expected_offset != size:
            raise FormatError(f"{path}: unexpected trailing or missing bytes")

    return Layout(
        path,
        size,
        magic,
        num_tensors,
        config,
        tuple(tensors),
    )


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
    tensors: tuple[Tensor, ...],
    default_bits: int,
    rules: list[Rule],
) -> list[int | None]:
    matches = [0] * len(rules)
    plan: list[int | None] = []
    for tensor in tensors:
        choice: int | None = default_bits
        for rule_index, rule in enumerate(rules):
            if fnmatch.fnmatchcase(tensor.name, rule.pattern):
                choice = rule.bits
                matches[rule_index] += 1
        plan.append(choice)
    for rule, count in zip(rules, matches, strict=True):
        if count == 0:
            raise FormatError(f"rule matched no tensors: {rule.pattern}")
    return plan


def packed_index(
    tensors: tuple[Tensor, ...],
    plan: list[int | None],
) -> tuple[bytes, int]:
    data_offset = align(
        HEADER.size + CONFIG_SIZE + len(tensors) * INDEX.size,
        2,
    )
    cursor = data_offset
    entries: list[bytes] = []
    for tensor, bits in zip(tensors, plan, strict=True):
        tag = RAW_INT64_TAG if bits is None else bits
        entries.append(
            INDEX.pack(
                tensor.raw_name,
                tensor.num_elements,
                cursor,
                tensor.ndims,
                tensor.shape0,
                tensor.shape1,
                tag,
            )
        )
        cursor += tensor.num_elements * (8 if bits is None else 2)
    return b"".join(entries), cursor


def canonical_mgw_index(
    tensors: tuple[Tensor, ...],
) -> tuple[bytes, int]:
    data_offset = align(
        HEADER.size + CONFIG_SIZE + len(tensors) * INDEX.size,
        8,
    )
    cursor = data_offset
    entries: list[bytes] = []
    for tensor in tensors:
        entries.append(
            INDEX.pack(
                tensor.raw_name,
                tensor.num_elements,
                cursor,
                tensor.ndims,
                tensor.shape0,
                tensor.shape1,
                0,
            )
        )
        cursor += tensor.num_elements * 8
    return b"".join(entries), cursor


def write_prefix(
    fd: int,
    magic: bytes,
    num_tensors: int,
    data_offset: int,
    config: bytes,
    raw_index: bytes,
) -> None:
    header = HEADER.pack(
        magic,
        FORMAT_VERSION,
        ENDIAN_TAG,
        num_tensors,
        HEADER.size + CONFIG_SIZE,
        data_offset,
        bytes(32),
    )
    os.pwrite(fd, header, 0)
    os.pwrite(fd, config, HEADER.size)
    os.pwrite(fd, raw_index, HEADER.size + CONFIG_SIZE)


def pack_mgwi(
    source: Path,
    destination: Path,
    default_bits: int,
    rules: list[Rule],
) -> None:
    if destination.exists():
        raise FormatError(f"refusing to overwrite {destination}")
    layout = read_layout(source, MGW_MAGIC)
    plan = choose_plan(layout.tensors, default_bits, rules)
    raw_index, output_size = packed_index(layout.tensors, plan)
    data_offset = align(
        HEADER.size + CONFIG_SIZE + layout.num_tensors * INDEX.size,
        2,
    )
    fd = os.open(destination, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    completed = False
    try:
        os.ftruncate(fd, output_size)
        write_prefix(
            fd,
            MGWI_MAGIC,
            layout.num_tensors,
            data_offset,
            layout.config,
            raw_index,
        )
        packed_layout = read_layout(destination, MGWI_MAGIC)
        for source_tensor, packed_tensor, bits in zip(
            layout.tensors,
            packed_layout.tensors,
            plan,
            strict=True,
        ):
            source_values = np.memmap(
                source,
                dtype="<i8",
                mode="r",
                offset=source_tensor.data_offset,
                shape=(source_tensor.num_elements,),
            )
            if bits is None:
                destination_values = np.memmap(
                    destination,
                    dtype="<i8",
                    mode="r+",
                    offset=packed_tensor.data_offset,
                    shape=(packed_tensor.num_elements,),
                )
                destination_values[:] = source_values
            else:
                quantum = 1 << (FP_FRACTION_BITS - bits)
                destination_values = np.memmap(
                    destination,
                    dtype="<i2",
                    mode="r+",
                    offset=packed_tensor.data_offset,
                    shape=(packed_tensor.num_elements,),
                )
                for start in range(0, source_tensor.num_elements, CHUNK_ELEMENTS):
                    end = min(
                        start + CHUNK_ELEMENTS,
                        source_tensor.num_elements,
                    )
                    values = np.asarray(source_values[start:end])
                    if np.any(values % quantum):
                        raise FormatError(
                            f"{source_tensor.name}: source is not on F{bits} grid"
                        )
                    scaled = values // quantum
                    minimum = int(scaled.min(initial=0))
                    maximum = int(scaled.max(initial=0))
                    if minimum < -32768 or maximum > 32767:
                        raise FormatError(
                            f"{source_tensor.name}: F{bits} scaled range "
                            f"{minimum}..{maximum} does not fit int16"
                        )
                    destination_values[start:end] = scaled
            destination_values.flush()
            del destination_values
            del source_values
        os.fsync(fd)
        completed = True
    finally:
        os.close(fd)
        if not completed:
            destination.unlink(missing_ok=True)


def unpack_mgwi(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FormatError(f"refusing to overwrite {destination}")
    layout = read_layout(source, MGWI_MAGIC)
    raw_index, output_size = canonical_mgw_index(layout.tensors)
    data_offset = align(
        HEADER.size + CONFIG_SIZE + layout.num_tensors * INDEX.size,
        8,
    )
    fd = os.open(destination, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    completed = False
    try:
        os.ftruncate(fd, output_size)
        write_prefix(
            fd,
            MGW_MAGIC,
            layout.num_tensors,
            data_offset,
            layout.config,
            raw_index,
        )
        unpacked_layout = read_layout(destination, MGW_MAGIC)
        for packed_tensor, output_tensor in zip(
            layout.tensors,
            unpacked_layout.tensors,
            strict=True,
        ):
            output_values = np.memmap(
                destination,
                dtype="<i8",
                mode="r+",
                offset=output_tensor.data_offset,
                shape=(output_tensor.num_elements,),
            )
            if packed_tensor.tag == RAW_INT64_TAG:
                packed_values = np.memmap(
                    source,
                    dtype="<i8",
                    mode="r",
                    offset=packed_tensor.data_offset,
                    shape=(packed_tensor.num_elements,),
                )
                output_values[:] = packed_values
            else:
                bits = packed_tensor.tag
                quantum = 1 << (FP_FRACTION_BITS - bits)
                packed_values = np.memmap(
                    source,
                    dtype="<i2",
                    mode="r",
                    offset=packed_tensor.data_offset,
                    shape=(packed_tensor.num_elements,),
                )
                for start in range(0, packed_tensor.num_elements, CHUNK_ELEMENTS):
                    end = min(
                        start + CHUNK_ELEMENTS,
                        packed_tensor.num_elements,
                    )
                    scaled = np.asarray(
                        packed_values[start:end],
                        dtype=np.int64,
                    )
                    output_values[start:end] = scaled * quantum
            output_values.flush()
            del packed_values
            del output_values
        os.fsync(fd)
        completed = True
    finally:
        os.close(fd)
        if not completed:
            destination.unlink(missing_ok=True)


def signed_bits_needed(minimum: int, maximum: int) -> int:
    for bits in range(1, 65):
        if minimum >= -(1 << (bits - 1)) and maximum <= (1 << (bits - 1)) - 1:
            return bits
    raise FormatError(f"range {minimum}..{maximum} exceeds signed int64")


def analyze_widths(source: Path) -> None:
    layout = read_layout(source, MGWI_MAGIC)
    tensor_counts: dict[int, int] = {}
    element_counts: dict[int, int] = {}
    ideal_payload_bytes = 0
    widest: list[tuple[str, int, int, int]] = []
    for tensor in layout.tensors:
        dtype = "<i8" if tensor.tag == RAW_INT64_TAG else "<i2"
        values = np.memmap(
            source,
            dtype=dtype,
            mode="r",
            offset=tensor.data_offset,
            shape=(tensor.num_elements,),
        )
        minimum = 0
        maximum = 0
        for start in range(0, tensor.num_elements, CHUNK_ELEMENTS):
            end = min(start + CHUNK_ELEMENTS, tensor.num_elements)
            chunk = values[start:end]
            minimum = min(minimum, int(chunk.min(initial=0)))
            maximum = max(maximum, int(chunk.max(initial=0)))
        del values
        bits = signed_bits_needed(minimum, maximum)
        tensor_counts[bits] = tensor_counts.get(bits, 0) + 1
        element_counts[bits] = (
            element_counts.get(bits, 0) + tensor.num_elements
        )
        ideal_payload_bytes += (tensor.num_elements * bits + 7) // 8
        widest.append((tensor.name, bits, minimum, maximum))

    metadata_bytes = align(
        HEADER.size + CONFIG_SIZE + layout.num_tensors * INDEX.size,
        2,
    )
    print(f"path={layout.path}")
    print(f"bytes={layout.size}")
    print(f"metadata_bytes={metadata_bytes}")
    print(f"int16_payload_bytes={layout.size - metadata_bytes}")
    print(f"ideal_tensor_bitpacked_payload_bytes={ideal_payload_bytes}")
    print(f"ideal_tensor_bitpacked_total_bytes={metadata_bytes + ideal_payload_bytes}")
    for bits in sorted(tensor_counts):
        print(
            f"B{bits} tensors={tensor_counts[bits]} "
            f"elements={element_counts[bits]}"
        )
    max_bits = max(tensor_counts)
    print(f"widest_signed_bits={max_bits}")
    for name, bits, minimum, maximum in widest:
        if bits == max_bits:
            print(
                f"widest_tensor={name} bits={bits} "
                f"range={minimum}..{maximum}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--default-bits", type=int, required=True)
    pack_parser.add_argument(
        "--set",
        dest="rules",
        action="append",
        type=parse_set_rule,
        default=[],
        metavar="GLOB=BITS",
    )
    pack_parser.add_argument(
        "--raw",
        dest="raw_rules",
        action="append",
        default=[],
        metavar="GLOB",
    )
    pack_parser.add_argument("source", type=Path)
    pack_parser.add_argument("destination", type=Path)

    unpack_parser = subparsers.add_parser("unpack")
    unpack_parser.add_argument("source", type=Path)
    unpack_parser.add_argument("destination", type=Path)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("source", type=Path)

    analyze_parser = subparsers.add_parser("analyze-widths")
    analyze_parser.add_argument("source", type=Path)

    args = parser.parse_args()
    if args.command == "pack":
        if not 0 <= args.default_bits <= FP_FRACTION_BITS:
            raise FormatError("--default-bits must be in [0, 48]")
        rules = list(args.rules)
        rules.extend(Rule(pattern, None) for pattern in args.raw_rules)
        pack_mgwi(
            args.source,
            args.destination,
            args.default_bits,
            rules,
        )
        print(
            f"packed_bytes={args.destination.stat().st_size} "
            f"source_bytes={args.source.stat().st_size}"
        )
    elif args.command == "unpack":
        unpack_mgwi(args.source, args.destination)
        print(
            f"unpacked_bytes={args.destination.stat().st_size} "
            f"packed_bytes={args.source.stat().st_size}"
        )
    elif args.command == "inspect":
        layout = read_layout(args.source, MGWI_MAGIC)
        counts: dict[str, int] = {}
        elements: dict[str, int] = {}
        for tensor in layout.tensors:
            label = (
                "raw64"
                if tensor.tag == RAW_INT64_TAG
                else f"F{tensor.tag}"
            )
            counts[label] = counts.get(label, 0) + 1
            elements[label] = elements.get(label, 0) + tensor.num_elements
        print(f"path={layout.path}")
        print(f"bytes={layout.size}")
        print(f"tensors={layout.num_tensors}")
        for label in sorted(counts):
            print(
                f"{label} tensors={counts[label]} "
                f"elements={elements[label]}"
            )
    else:
        analyze_widths(args.source)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormatError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

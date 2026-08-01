#!/usr/bin/env python3
"""Rebuild the full uniform-F12 MicroGPT host and RP2040 fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_WIDE_SHA256 = (
    "742cbd6d0b750bf3d164a23d97390171e3fe545ee9d87a2b0e843d3d8d1ae9f4"
)
LANE_RE = re.compile(
    r"^LANE (wide|packed|corrupt) sample_hash=([0-9a-f]{16}) "
    r"logits_hash=([0-9a-f]{16}) steps=([0-9]+)$",
    re.MULTILINE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_or_refresh(path: Path, data: bytes, refresh: bool) -> None:
    if path.exists() and path.read_bytes() == data:
        return
    if not refresh:
        state = "stale" if path.exists() else "missing"
        raise RuntimeError(
            f"generated fixture is {state}: {path}; rerun with --refresh"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def load_pack_module(repo: Path):
    module_path = repo / "tools" / "mgwi_pack.py"
    spec = importlib.util.spec_from_file_location("mgwi_pack", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load mgwi_pack.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def c_header(symbol: str, include_macro: str, data: bytes, digest: str) -> bytes:
    guard = f"INT_LLM_{symbol.upper()}_H"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "#include <stdint.h>",
        f"#define {symbol.upper()}_BYTES {len(data)}u",
        f'#define {symbol.upper()}_SHA256 "{digest}"',
        f"#if {include_macro}",
        f"__attribute__((aligned(8))) static const uint8_t {symbol}[] = {{",
    ]
    for offset in range(0, len(data), 16):
        lines.append(
            "    " + "".join(f"0x{byte:02x}," for byte in data[offset : offset + 16])
        )
    lines.extend(
        [
            "};",
            f"static const size_t {symbol}_len = sizeof({symbol});",
            "#endif",
            f"#endif /* {guard} */",
            "",
        ]
    )
    return "\n".join(lines).encode()


def parse_lanes(output: str) -> dict[str, dict[str, int]]:
    lanes: dict[str, dict[str, int]] = {}
    for name, sample_hash, logits_hash, steps in LANE_RE.findall(output):
        lanes[name] = {
            "sample_hash": int(sample_hash, 16),
            "logits_hash": int(logits_hash, 16),
            "steps": int(steps),
        }
    if set(lanes) != {"wide", "packed", "corrupt"}:
        raise RuntimeError("host transcript is missing lane records")
    if "HOST_RESULT positive=PASS negative=PASS" not in output:
        raise RuntimeError("host differential did not pass")
    return lanes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify or explicitly refresh the uniform-F12 fixture",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rewrite generated fixture files deliberately",
    )
    args = parser.parse_args()

    spike = Path(__file__).resolve().parent
    repo = spike.parents[1]
    models = spike / "models"
    harness = spike / "harness"
    wide_path = models / "uniform-f12.mgw"
    wide_raw = wide_path.read_bytes()
    if sha256_bytes(wide_raw) != EXPECTED_WIDE_SHA256:
        raise RuntimeError("uniform-F12 source identity mismatch")
    packed_path = models / "uniform-f12.mgwi"
    corrupt_path = models / "uniform-f12-corrupt.mgwi"

    packer = load_pack_module(repo)
    with tempfile.TemporaryDirectory(prefix="microgpt-f12-") as temporary:
        temporary_path = Path(temporary)
        candidate = temporary_path / "uniform-f12.mgwi"
        unpacked = temporary_path / "roundtrip.mgw"
        packer.pack_mgwi(
            wide_path,
            candidate,
            12,
            [
                packer.Rule("tokenizer.uchars", None),
                packer.Rule("rng.state", None),
            ],
        )
        packer.unpack_mgwi(candidate, unpacked)
        if unpacked.read_bytes() != wide_raw:
            raise RuntimeError("packed round trip is not byte-identical")
        packed_raw = candidate.read_bytes()
    verify_or_refresh(packed_path, packed_raw, args.refresh)

    layout = packer.read_layout(packed_path, packer.MGWI_MAGIC)
    wte = next(tensor for tensor in layout.tensors if tensor.name == "wte")
    vocab_size = struct.unpack_from("<i", layout.config, 24)[0]
    hidden_dim = struct.unpack_from("<i", layout.config, 0)[0]
    corrupt_element = (vocab_size - 1) * hidden_dim
    if corrupt_element >= wte.num_elements:
        raise RuntimeError("corruption point is outside BOS embedding row")
    corrupt_offset = wte.data_offset + 2 * corrupt_element
    old_code = struct.unpack_from("<h", packed_raw, corrupt_offset)[0]
    new_code = old_code + 1 if old_code < 32767 else old_code - 1
    corrupt_raw = bytearray(packed_raw)
    struct.pack_into("<h", corrupt_raw, corrupt_offset, new_code)
    verify_or_refresh(corrupt_path, bytes(corrupt_raw), args.refresh)

    verify_or_refresh(
        harness / "src" / "microgpt_int.c",
        (repo / "src" / "microgpt_int.c").read_bytes(),
        args.refresh,
    )
    verify_or_refresh(
        harness / "include" / "fp_math.h",
        (repo / "src" / "fp_math.h").read_bytes(),
        args.refresh,
    )
    verify_or_refresh(
        harness / "include" / "wide_model.h",
        c_header("wide_model", "MGPT_INCLUDE_WIDE", wide_raw, sha256_bytes(wide_raw)),
        args.refresh,
    )
    verify_or_refresh(
        harness / "include" / "packed_model.h",
        c_header(
            "packed_model", "MGPT_INCLUDE_PACKED", packed_raw, sha256_bytes(packed_raw)
        ),
        args.refresh,
    )
    verify_or_refresh(
        harness / "include" / "corrupt_model.h",
        c_header(
            "corrupt_model",
            "MGPT_INCLUDE_CORRUPT",
            bytes(corrupt_raw),
            sha256_bytes(bytes(corrupt_raw)),
        ),
        args.refresh,
    )

    compiler = Path(shutil.which(os.environ.get("CC", "cc")) or "")
    if not compiler.is_file():
        raise RuntimeError("C compiler not found")
    common = [
        str(compiler),
        "-std=c11",
        "-O2",
        "-fwrapv",
        "-Wno-unused-function",
        "-Wno-unused-variable",
        "-DMGPT_NO_TRAIN",
        "-DMGPT_NO_MAIN",
        "-DMGPT_PACKED_F12",
        "-DMGPT_OBSERVE_LOGITS",
        str(spike / "hostref" / "host_main.c"),
        str(repo / "src" / "microgpt_int.c"),
    ]
    outputs: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="microgpt-f12-host-") as host_tmp:
        host_dir = Path(host_tmp)
        for backend, extra in (
            ("native", []),
            ("portable", ["-DFP_MATH_FORCE_PORTABLE"]),
        ):
            executable = host_dir / f"host_{backend}"
            command = common[:10] + extra + common[10:] + ["-o", str(executable)]
            subprocess.run(command, check=True)
            output = subprocess.run(
                [str(executable), str(wide_path), str(packed_path), str(corrupt_path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout
            outputs[backend] = output
    if outputs["native"] != outputs["portable"]:
        raise RuntimeError("native and portable host transcripts differ")
    lanes = parse_lanes(outputs["native"])

    pins = (
        "#ifndef INT_LLM_MICROGPT_F12_PINS_H\n"
        "#define INT_LLM_MICROGPT_F12_PINS_H\n"
        f"#define PIN_SAMPLE_HASH UINT64_C(0x{lanes['wide']['sample_hash']:016x})\n"
        f"#define PIN_LOGITS_HASH UINT64_C(0x{lanes['wide']['logits_hash']:016x})\n"
        f"#define PIN_STEPS {lanes['wide']['steps']}u\n"
        f"#define PIN_CORRUPT_SAMPLE_HASH UINT64_C(0x{lanes['corrupt']['sample_hash']:016x})\n"
        f"#define PIN_CORRUPT_LOGITS_HASH UINT64_C(0x{lanes['corrupt']['logits_hash']:016x})\n"
        f"#define PIN_CORRUPT_STEPS {lanes['corrupt']['steps']}u\n"
        "#endif\n"
    ).encode()
    verify_or_refresh(harness / "include" / "pins.h", pins, args.refresh)

    manifest = {
        "schema": "microgpt_f12_full_prepare_v3",
        "wide": {"bytes": len(wide_raw), "sha256": sha256_bytes(wide_raw)},
        "packed": {"bytes": len(packed_raw), "sha256": sha256_bytes(packed_raw)},
        "corrupt": {
            "bytes": len(corrupt_raw),
            "sha256": sha256_bytes(bytes(corrupt_raw)),
            "tensor": "wte",
            "element": corrupt_element,
            "old_code": old_code,
            "new_code": new_code,
        },
        "payload_saving_bytes": len(wide_raw) - len(packed_raw),
        "source": {
            "microgpt_int.c": sha256_file(repo / "src" / "microgpt_int.c"),
            "fp_math.h": sha256_file(repo / "src" / "fp_math.h"),
            "mgwi_pack.py": sha256_file(repo / "tools" / "mgwi_pack.py"),
            "host_main.c": sha256_file(spike / "hostref" / "host_main.c"),
        },
        "host_transcript_sha256": sha256_bytes(outputs["native"].encode()),
        "lanes": lanes,
    }
    verify_or_refresh(
        spike / "prepare-manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        args.refresh,
    )
    print(outputs["native"], end="")
    print(f"wide_bytes={len(wide_raw)} packed_bytes={len(packed_raw)}")
    print(f"payload_saving_bytes={len(wide_raw) - len(packed_raw)}")
    print(f"PREPARE_RESULT=PASS mode={'refresh' if args.refresh else 'check'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

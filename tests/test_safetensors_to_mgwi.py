#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spikes" / "safetensors-direct-mgwi"))

import run_gate as direct_gate  # noqa: E402
import run_correctness_gate as correctness_gate  # noqa: E402
from mgwi_pack import Rule as ReferenceRule  # noqa: E402
from mgwi_pack import pack_mgwi  # noqa: E402
from safetensors_to_mgwi import (  # noqa: E402
    FormatError,
    Rule,
    direct_pack,
    parse_json,
    q1648_to_int16_codes,
    source_chunk_to_q1648,
)


BF16_VALUES = (
    0x0000,
    0x8000,
    0x3F80,
    0xBF80,
    0x3F00,
    0xBF00,
    0x3900,
    0xB900,
    0x4000,
    0xC000,
)
F16_VALUES = (
    0x0000,
    0x8000,
    0x3C00,
    0xBC00,
    0x3800,
    0xB800,
    0x0800,
    0x8800,
    0x4000,
    0xC000,
    0x0001,
)
RULES = [
    Rule("model.layers.*.self_attn.*_proj.weight", 12),
    Rule("model.layers.*.mlp.*_proj.weight", 12),
]
REFERENCE_RULES = [
    ReferenceRule("model.layers.*.self_attn.*_proj.weight", 12),
    ReferenceRule("model.layers.*.mlp.*_proj.weight", 12),
]
TEST_HEADER = struct.Struct("<4sIIIQQ32s")
TEST_INDEX = struct.Struct("<64sQQIIII")


def canonical_specs() -> list[tuple[str, tuple[int, ...]]]:
    return [
        ("model.embed_tokens.weight", (4, 2)),
        ("model.norm.weight", (2,)),
        ("lm_head.weight", (4, 2)),
        ("model.layers.0.input_layernorm.weight", (2,)),
        ("model.layers.0.self_attn.q_proj.weight", (2, 2)),
        ("model.layers.0.self_attn.k_proj.weight", (2, 2)),
        ("model.layers.0.self_attn.v_proj.weight", (2, 2)),
        ("model.layers.0.self_attn.o_proj.weight", (2, 2)),
        ("model.layers.0.post_attention_layernorm.weight", (2,)),
        ("model.layers.0.mlp.gate_proj.weight", (3, 2)),
        ("model.layers.0.mlp.up_proj.weight", (3, 2)),
        ("model.layers.0.mlp.down_proj.weight", (2, 3)),
    ]


def write_synthetic_model(
    root: Path,
    *,
    overflow: bool = False,
    q1648_overflow: bool = False,
    huge_q1648_overflow: bool = False,
    nonfinite: bool = False,
    unsupported_f32: bool = False,
    extra_descriptor_field: bool = False,
    invalid_metadata: bool = False,
) -> Path:
    root.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "intermediate_size": 3,
        "vocab_size": 4,
        "max_position_embeddings": 8,
        "rope_theta": 10000.0,
    }
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )

    descriptors: dict[str, object] = {}
    payload = bytearray()
    for index, (name, shape) in enumerate(canonical_specs()):
        dtype = "F32" if unsupported_f32 and index == 0 else (
            "BF16" if index % 2 == 0 else "F16"
        )
        values = BF16_VALUES if dtype == "BF16" else F16_VALUES
        elements = int(np.prod(shape))
        start = len(payload)
        for element in range(elements):
            if dtype == "F32":
                payload.extend(struct.pack("<f", 1.0))
            else:
                raw = values[(index + element) % len(values)]
                if overflow and name == "model.embed_tokens.weight" and element == 0:
                    raw = 0x4180 if dtype == "BF16" else 0x4C00  # +16.0
                if (
                    q1648_overflow
                    and name == "model.embed_tokens.weight"
                    and element == 0
                ):
                    raw = 0x4700  # finite BF16 +32768.0 => Q16.48 overflow
                if (
                    huge_q1648_overflow
                    and name == "model.embed_tokens.weight"
                    and element == 0
                ):
                    raw = 0x6000  # finite BF16 with a Q16.48 shift above 63
                if (
                    nonfinite
                    and name == "model.embed_tokens.weight"
                    and element == 0
                ):
                    raw = 0x7F80  # BF16 positive infinity
                payload.extend(struct.pack("<H", raw))
        descriptors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
        if extra_descriptor_field and index == 0:
            descriptors[name]["unexpected"] = 1  # type: ignore[index]

    # Header order is deliberately different from canonical MGWI order.
    header_object = {
        "__metadata__": {"format": 1 if invalid_metadata else "pt"},
        **dict(reversed(list(descriptors.items()))),
    }
    raw_header = json.dumps(
        header_object,
        separators=(",", ":"),
    ).encode("utf-8")
    with (root / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw_header)))
        stream.write(raw_header)
        stream.write(payload)
    return root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def replace_first_tensor_offsets(model: Path, offsets: list[int]) -> None:
    path = model / "model.safetensors"
    raw = path.read_bytes()
    (header_size,) = struct.unpack_from("<Q", raw)
    header_end = 8 + header_size
    parsed = json.loads(raw[8:header_end])
    first_name = canonical_specs()[0][0]
    parsed[first_name]["data_offsets"] = offsets
    new_header = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(new_header)))
        stream.write(new_header)
        stream.write(raw[header_end:])


def files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(64 * 1024)
            right_chunk = right_stream.read(64 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


class DirectSafetensorsMgwiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        completed = subprocess.run(
            ["make", "llama-mgwi", "mgw_round"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout)

    def test_direct_output_matches_existing_three_stage_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model")
            oracle = temp / "oracle.mgw"
            rounded = temp / "rounded.mgw"
            expected = temp / "expected.mgwi"
            direct = temp / "direct.mgwi"

            exported = subprocess.run(
                [
                    str(ROOT / "llama_mgwi"),
                    str(model),
                    "--export-native",
                    str(oracle),
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(exported.returncode, 0, exported.stdout)
            rounded_run = subprocess.run(
                [
                    str(ROOT / "mgw_round"),
                    "--default-bits",
                    "11",
                    "--set",
                    "model.layers.*.self_attn.*_proj.weight=12",
                    "--set",
                    "model.layers.*.mlp.*_proj.weight=12",
                    str(oracle),
                    str(rounded),
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(rounded_run.returncode, 0, rounded_run.stdout)
            pack_mgwi(rounded, expected, 11, REFERENCE_RULES)

            with mock.patch("safetensors_to_mgwi.CHUNK_ELEMENTS", 3):
                result = direct_pack(
                    model,
                    direct,
                    11,
                    RULES,
                    expected_source_sha256=file_sha256(model / "model.safetensors"),
                    expected_config_sha256=file_sha256(model / "config.json"),
                    expected_output_sha256=file_sha256(expected),
                )
            self.assertTrue(files_equal(direct, expected))
            self.assertEqual(result["num_tensors"], 12)
            self.assertEqual(result["num_elements"], 56)
            self.assertEqual(result["output_sha256"], file_sha256(expected))
            self.assertEqual(
                result["config_sha256"],
                file_sha256(model / "config.json"),
            )

            # Independent format check: do not use either serializer to parse
            # the candidate's header/index/payload geometry.
            with direct.open("rb") as stream:
                raw_header = stream.read(TEST_HEADER.size)
                magic, version, endian, count, index_offset, data_offset, _ = (
                    TEST_HEADER.unpack(raw_header)
                )
                self.assertEqual((magic, version, endian, count), (b"MGWI", 1, 0x01020304, 12))
                self.assertEqual(index_offset, TEST_HEADER.size + 64)
                expected_data_offset = (
                    index_offset + count * TEST_INDEX.size + 1
                ) & ~1
                self.assertEqual(data_offset, expected_data_offset)
                stream.seek(index_offset)
                cursor = data_offset
                tags = []
                for name, shape in canonical_specs():
                    entry = TEST_INDEX.unpack(stream.read(TEST_INDEX.size))
                    raw_name, elements, offset, ndims, shape0, shape1, tag = entry
                    self.assertEqual(raw_name.split(b"\0", 1)[0].decode(), name)
                    self.assertEqual(elements, int(np.prod(shape)))
                    self.assertEqual(offset, cursor)
                    self.assertEqual((ndims, shape0, shape1), (len(shape), shape[0], shape[1] if len(shape) == 2 else 0))
                    tags.append(tag)
                    cursor += elements * 2
                self.assertEqual(cursor, direct.stat().st_size)
            self.assertEqual(tags.count(12), 7)
            self.assertEqual(tags.count(11), 5)

            with mock.patch.multiple(
                direct_gate,
                PACKED_BYTES=direct.stat().st_size,
                PACKED_SHA256=file_sha256(direct),
                WIDE_BYTES=rounded.stat().st_size,
                WIDE_SHA256=file_sha256(rounded),
                EXPECTED_TENSORS=12,
                EXPECTED_ELEMENTS=56,
                EXPECTED_F11_TENSORS=5,
                EXPECTED_F11_ELEMENTS=22,
                EXPECTED_F12_TENSORS=7,
                EXPECTED_F12_ELEMENTS=34,
                COMPARE_BYTES=17,
                COMPARE_ELEMENTS=3,
            ):
                full = direct_gate.compare_files(direct, expected)
                unpacked = direct_gate.compare_unpacked(direct, rounded)
            self.assertTrue(full["byte_identical"])
            self.assertEqual(full["bytes_compared"], direct.stat().st_size)
            self.assertTrue(unpacked["unpacked_byte_identical"])
            self.assertEqual(unpacked["elements_compared"], 56)

    def test_source_hash_mismatch_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model")
            output = temp / "candidate.mgwi"
            with self.assertRaisesRegex(FormatError, "source SHA-256 mismatch"):
                direct_pack(model, output, 11, RULES, "0" * 64)
            self.assertFalse(output.exists())

    def test_config_hash_mismatch_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model")
            output = temp / "candidate.mgwi"
            with self.assertRaisesRegex(FormatError, "config SHA-256 mismatch"):
                direct_pack(
                    model,
                    output,
                    11,
                    RULES,
                    expected_config_sha256="0" * 64,
                )
            self.assertFalse(output.exists())

    def test_out_of_range_code_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model", overflow=True)
            output = temp / "candidate.mgwi"
            with self.assertRaisesRegex(FormatError, "does not fit int16"):
                direct_pack(model, output, 11, RULES)
            self.assertFalse(output.exists())

    def test_q1648_overflow_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model", q1648_overflow=True)
            output = temp / "candidate.mgwi"
            with self.assertRaisesRegex(FormatError, "overflows Q16.48"):
                direct_pack(model, output, 11, RULES)
            self.assertFalse(output.exists())

    def test_huge_q1648_overflow_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(
                temp / "model",
                huge_q1648_overflow=True,
            )
            output = temp / "candidate.mgwi"
            with self.assertRaisesRegex(FormatError, "overflows Q16.48"):
                direct_pack(model, output, 11, RULES)
            self.assertFalse(output.exists())

    def test_nonfinite_source_removes_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model", nonfinite=True)
            output = temp / "candidate.mgwi"
            with self.assertRaisesRegex(FormatError, "non-finite"):
                direct_pack(model, output, 11, RULES)
            self.assertFalse(output.exists())

    def test_unsupported_f32_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model", unsupported_f32=True)
            output = temp / "candidate.mgwi"
            with self.assertRaisesRegex(FormatError, "unsupported dtype 'F32'"):
                direct_pack(model, output, 11, RULES)
            self.assertFalse(output.exists())

    def test_c_reference_rejects_wrapping_negative_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            model = write_synthetic_model(temp / "model")
            replace_first_tensor_offsets(model, [-1, 15])
            output = temp / "candidate.mgw"
            completed = subprocess.run(
                [
                    str(ROOT / "llama_mgwi"),
                    str(model),
                    "--export-native",
                    str(output),
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertFalse(output.exists())

    def test_nonstandard_descriptor_and_metadata_are_rejected(self) -> None:
        for option, message in (
            ({"extra_descriptor_field": True}, "unexpected descriptor fields"),
            ({"invalid_metadata": True}, "metadata__ values must be strings"),
        ):
            with self.subTest(option=option):
                with tempfile.TemporaryDirectory() as temporary:
                    temp = Path(temporary)
                    model = write_synthetic_model(temp / "model", **option)
                    with self.assertRaisesRegex(FormatError, message):
                        direct_pack(model, temp / "candidate.mgwi", 11, RULES)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(FormatError, "duplicate JSON key"):
            parse_json(b'{"dtype":"BF16","dtype":"F16"}', "fixture")

    def test_non_json_numeric_constants_are_rejected(self) -> None:
        with self.assertRaisesRegex(FormatError, "non-JSON numeric constant"):
            parse_json(b'{"rope_theta":NaN}', "fixture")

    def test_known_source_decoding_and_ties_away_from_zero(self) -> None:
        bf16 = np.array([0x3F80, 0xBF80, 0x3900, 0xB900], dtype="<u2")
        q_values = source_chunk_to_q1648(bf16, "BF16", "fixture")
        codes = q1648_to_int16_codes(q_values, 12, "fixture")
        self.assertEqual(codes.tolist(), [4096, -4096, 1, -1])

        f16_subnormal = np.array([0x0001, 0x8001], dtype="<u2")
        q_subnormal = source_chunk_to_q1648(
            f16_subnormal,
            "F16",
            "fixture",
        )
        self.assertEqual(q_subnormal.tolist(), [1 << 24, -(1 << 24)])
        self.assertEqual(
            q1648_to_int16_codes(q_subnormal, 12, "fixture").tolist(),
            [0, 0],
        )

    def test_correctness_gate_parsers_require_exact_sentinels(self) -> None:
        benchmark = (
            "  TOTAL: 80/80 tokens match (100.0%)\n"
            "RAW_LOGITS_FNV64=2a32782dc6e68e12 vectors=102\n"
        )
        parsed = correctness_gate.parse_benchmark(
            benchmark,
            "fixture",
            correctness_gate.RAW_FNV64,
        )
        self.assertEqual(parsed["matched_decisions"], 80)
        for corrupted in (
            benchmark.replace("80/80", "79/80", 1),
            benchmark.replace("2a32782dc6e68e12", "0" * 16),
            benchmark + "RAW_LOGITS_FNV64=2a32782dc6e68e12 vectors=102\n",
        ):
            with self.assertRaises(correctness_gate.CorrectnessError):
                correctness_gate.parse_benchmark(
                    corrupted,
                    "fixture",
                    correctness_gate.RAW_FNV64,
                )

    def test_correctness_gate_generation_parser_rejects_partial_or_duplicate(self) -> None:
        generation = (
            "  Generated 20 tokens in 1.00 seconds\n"
            "  Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "
            "11, 12, 13, 14, 15, 16, 17, 18, 19, 20]\n"
        )
        parsed = correctness_gate.parse_generation(generation, "fixture")
        self.assertEqual(parsed.tokens, tuple(range(1, 21)))
        self.assertEqual(parsed.emitted_tokens, 20)
        self.assertEqual(parsed.evaluated_decisions, 20)
        early = correctness_gate.parse_generation(
            generation.replace("Generated 20", "Generated 19"),
            "fixture",
        )
        self.assertEqual(early.emitted_tokens, 19)
        self.assertEqual(early.evaluated_decisions, 20)
        for corrupted in (
            generation.replace("Generated 20", "Generated 0"),
            generation.replace("Generated 20", "Generated 21"),
            generation + "  Tokens: [1, 2, 3]\n",
        ):
            with self.assertRaises(correctness_gate.CorrectnessError):
                correctness_gate.parse_generation(corrupted, "fixture")


if __name__ == "__main__":
    unittest.main()

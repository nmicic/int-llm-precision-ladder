#!/usr/bin/env python3

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mgw_precision import (  # noqa: E402
    parse_mgw,
    quantize_mgw,
    quantize_mgw_plan,
    round_q1648,
    sha256,
    verify_candidate,
    verify_mgw_plan,
)


class PrecisionTests(unittest.TestCase):
    def test_round_ties_away_from_zero(self) -> None:
        self.assertEqual(round_q1648(0, 47), (0, False))
        self.assertEqual(round_q1648(1, 47), (2, False))
        self.assertEqual(round_q1648(-1, 47), (-2, False))
        self.assertEqual(round_q1648(2, 47), (2, False))
        self.assertEqual(round_q1648(3, 47), (4, False))
        self.assertEqual(round_q1648(-3, 47), (-4, False))

    def test_committed_model_layout(self) -> None:
        info = parse_mgw(ROOT / "model.mgw")
        weights = [tensor for tensor in info.tensors if tensor.is_weight]
        metadata = [tensor for tensor in info.tensors if not tensor.is_weight]
        self.assertEqual(info.vocab_size, 27)
        self.assertEqual(info.hidden_dim, 32)
        self.assertEqual(info.num_layers, 1)
        self.assertEqual(len(weights), 9)
        self.assertEqual(sum(tensor.num_elements for tensor in weights), 14272)
        self.assertEqual(
            {tensor.name for tensor in metadata},
            {"tokenizer.uchars", "rng.state"},
        )

    def test_f48_is_byte_identical(self) -> None:
        source = ROOT / "model.mgw"
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "f48.mgw"
            stats = quantize_mgw(source, candidate, 48)
            verify_candidate(source, candidate, 48)
            self.assertEqual(stats.changed_weights, 0)
            self.assertEqual(sha256(source), sha256(candidate))

    def test_coarse_candidate_is_on_grid(self) -> None:
        source = ROOT / "model.mgw"
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "f24.mgw"
            stats = quantize_mgw(source, candidate, 24)
            verify_candidate(source, candidate, 24)
            self.assertGreater(stats.changed_weights, 0)
            self.assertEqual(stats.saturations, 0)

    def test_selected_tensor_leaves_every_other_tensor_unchanged(self) -> None:
        source = ROOT / "model.mgw"
        info = parse_mgw(source)
        source_raw = source.read_bytes()
        selected = {"wpe"}
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "wpe-f9.mgw"
            stats = quantize_mgw(
                source,
                candidate,
                9,
                tensor_names=selected,
            )
            verify_candidate(
                source,
                candidate,
                9,
                tensor_names=selected,
            )
            candidate_raw = candidate.read_bytes()
            self.assertEqual(stats.total_weights, 8 * 32)
            for tensor in info.tensors:
                if tensor.name in selected:
                    continue
                start = tensor.data_offset
                end = start + tensor.byte_size
                self.assertEqual(source_raw[start:end], candidate_raw[start:end])

    def test_streaming_c_converter_matches_python_mixed_plan(self) -> None:
        source = ROOT / "model.mgw"
        converter = ROOT / "mgw_round"
        plan = {
            "lm_head": 12,
            "tokenizer.uchars": 48,
            "rng.state": 48,
        }
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            expected = temp_root / "python.mgw"
            actual = temp_root / "c.mgw"
            quantize_mgw_plan(source, expected, 9, plan)
            verify_mgw_plan(source, expected, 9, plan)
            completed = subprocess.run(
                [
                    str(converter),
                    "--default-bits",
                    "9",
                    "--set",
                    "lm_head=12",
                    "--set",
                    "tokenizer.uchars=48",
                    "--set",
                    "rng.state=48",
                    str(source),
                    str(actual),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertEqual(sha256(expected), sha256(actual))

    def test_streaming_c_converter_f48_is_byte_identical(self) -> None:
        source = ROOT / "model.mgw"
        converter = ROOT / "mgw_round"
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "f48.mgw"
            completed = subprocess.run(
                [
                    str(converter),
                    "--default-bits",
                    "48",
                    str(source),
                    str(candidate),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertEqual(sha256(source), sha256(candidate))

    def test_streaming_c_converter_analyze_only_reports_scaled_range(self) -> None:
        source = ROOT / "model.mgw"
        converter = ROOT / "mgw_round"
        completed = subprocess.run(
            [
                str(converter),
                "--analyze-only",
                "--default-bits",
                "12",
                str(source),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("mode=analyze-only", completed.stdout)
        self.assertIn("scaled_min=", completed.stdout)
        self.assertIn("scaled_max=", completed.stdout)

    def test_mgwi_int16_pack_round_trip_is_byte_identical(self) -> None:
        source = ROOT / "model.mgw"
        packer = ROOT / "tools" / "mgwi_pack.py"
        plan = {
            "lm_head": 12,
            "tokenizer.uchars": 48,
            "rng.state": 48,
        }
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            rounded = temp_root / "rounded.mgw"
            packed = temp_root / "rounded.mgwi"
            unpacked = temp_root / "unpacked.mgw"
            quantize_mgw_plan(source, rounded, 9, plan)
            pack = subprocess.run(
                [
                    sys.executable,
                    str(packer),
                    "pack",
                    "--default-bits",
                    "9",
                    "--set",
                    "lm_head=12",
                    "--set",
                    "tokenizer.uchars=48",
                    "--raw",
                    "rng.state",
                    str(rounded),
                    str(packed),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(pack.returncode, 0, pack.stdout)
            analyze = subprocess.run(
                [
                    sys.executable,
                    str(packer),
                    "analyze-widths",
                    str(packed),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(analyze.returncode, 0, analyze.stdout)
            self.assertIn(
                "ideal_tensor_bitpacked_total_bytes=",
                analyze.stdout,
            )
            unpack = subprocess.run(
                [
                    sys.executable,
                    str(packer),
                    "unpack",
                    str(packed),
                    str(unpacked),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(unpack.returncode, 0, unpack.stdout)
            self.assertEqual(sha256(rounded), sha256(unpacked))
            self.assertLess(packed.stat().st_size, rounded.stat().st_size)


if __name__ == "__main__":
    unittest.main()

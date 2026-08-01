#!/usr/bin/env python3
"""Regression tests for actual decision accounting in the shell gate."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tools" / "run_heldout_gate.sh"


FAKE_MODEL = """#!/bin/sh
model=$1
shift
prompt=
while [ \"$#\" -gt 0 ]; do
    if [ \"$1\" = \"--prompt\" ]; then
        prompt=$2
        shift 2
    else
        shift
    fi
done

case $prompt in
    full)
        generated=20
        tokens='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'
        ;;
    early)
        generated=17
        tokens='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]'
        ;;
    *)
        exit 9
        ;;
esac

case $model in
    *candidate-bad*)
        generated=16
        tokens='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]'
        ;;
esac

printf '  Generated %s tokens in 1.0 seconds\\n' \"$generated\"
printf '  Tokens: %s\\n' \"$tokens\"
"""


class HeldoutGateAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "fake-model.sh"
        self.binary.write_text(FAKE_MODEL, encoding="utf-8")
        self.binary.chmod(0o700)
        self.exact = self.root / "exact.mgw"
        self.candidate = self.root / "candidate.mgw"
        self.exact.write_bytes(b"exact")
        self.candidate.write_bytes(b"candidate")
        self.ref_dir = self.root / "ref"
        self.ref_dir.mkdir()
        self.prompts = self.root / "prompts.tsv"
        self.prompts.write_text("full\tfull\nearly\tearly\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_gate(
        self,
        candidate: Path,
        label: str,
        *,
        path: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result_dir = self.root / f"result-{label}"
        environment = dict(os.environ)
        environment["LC_ALL"] = "C"
        if path is not None:
            environment["PATH"] = path
        return subprocess.run(
            [
                "/bin/sh",
                str(GATE),
                str(self.binary),
                str(self.exact),
                str(candidate),
                str(self.ref_dir),
                str(self.prompts),
                str(result_dir),
                label,
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_early_eos_counts_one_evaluated_decision(self) -> None:
        completed = self.run_gate(self.candidate, "pass")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("early: PASS (emitted=17, evaluated_decisions=18)", completed.stdout)
        self.assertIn(
            "emitted=37, matched_decisions=38, maximum_requested=40",
            completed.stdout,
        )
        self.assertNotIn("20/20 generated", completed.stdout)

    def test_different_generated_count_fails(self) -> None:
        bad = self.root / "candidate-bad.mgw"
        bad.write_bytes(b"candidate-bad")
        completed = self.run_gate(bad, "fail")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("FAIL (generated exact=20 candidate=16)", completed.stdout)

    def test_shasum_fallback_without_sha256sum(self) -> None:
        shasum = shutil.which("shasum")
        if shasum is None:
            self.skipTest("shasum is unavailable")
        fallback_bin = self.root / "fallback-bin"
        fallback_bin.mkdir()
        for command in ("awk", "cmp", "mkdir", "tee"):
            source = shutil.which(command)
            self.assertIsNotNone(source, command)
            (fallback_bin / command).symlink_to(source)
        (fallback_bin / "shasum").symlink_to(shasum)

        completed = self.run_gate(
            self.candidate,
            "shasum",
            path=str(fallback_bin),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = (
            self.root / "result-shasum" / "shasum-summary.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("# binary_sha256 ", summary)


if __name__ == "__main__":
    unittest.main()

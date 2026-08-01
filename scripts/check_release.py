#!/usr/bin/env python3
"""Validate the self-contained publication tree without external execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "LICENSE",
    "README.md",
    "RESULTS.md",
    "TINYLLAMA_RESULTS.md",
    "MCU_RESULTS.md",
    "GPU_RESULTS.md",
    "PROVENANCE.md",
    "VALIDATION.md",
    "Makefile",
    "model.mgw",
    "src/fp_math.h",
    "src/microgpt_int.c",
    "src/llama/fp_math.h",
    "src/llama/llama_int.c",
    "src/llama/safetensors.h",
    "src/llama/tokenizer.h",
    "tools/README.md",
    "tools/mgw_precision.py",
    "tools/mgw_round.c",
    "tools/mgwi_pack.py",
    "tools/safetensors_to_mgwi.py",
    "tests/test_precision.py",
    "tests/test_run_heldout_gate.py",
    "tests/test_safetensors_conversion.c",
    "tests/test_safetensors_to_mgwi.py",
    "results/tinyllama_summary.json",
    "spikes/mcu-f12-microgpt/models/uniform-f12.mgw",
    "spikes/mcu-f12-microgpt/models/uniform-f12.mgwi",
    "spikes/mcu-f12-microgpt/models/uniform-f12-corrupt.mgwi",
    "spikes/mcu-f12-microgpt/prepare-manifest.json",
    "spikes/safetensors-direct-mgwi/results/2026-07-31-amd-x86_64/gate-result.json",
    "spikes/safetensors-direct-mgwi/results/2026-07-31-amd-x86_64/correctness-result.json",
)

ABSENT = (
    "AGENTS.md",
    "BOOTSTRAP.md",
    "CLAUDE.md",
    "GEMINI.md",
    "DECISIONS.md",
    "OPEN.md",
    "PROJECT_CONTEXT.md",
    "INTERNAL_RELEASE_BOUNDARY.md",
    "reviews",
    "decisions",
    "prompts",
    "internal-archive",
)

PINNED_SHA256 = {
    "model.mgw": "466cfe9dba7b888cdaa23dedf4b10351826795793448c8e95dcb0f7a61ed33eb",
    "src/fp_math.h": "21d6952213c1afcd1ad22e9b7a630cfdc619d10f7497978d00a2a9a0d9d95735",
    "src/microgpt_int.c": "b25fc25ec2c7f0f6fd9511412f53d46473cc6a89e26d14af4d32b9ba8a0021d4",
    "src/llama/llama_int.c": "5908bf5bcd224ac8528e21fb113ff466ea497643f6248e67ce343545c01a560c",
    "src/llama/fp_math.h": "21d6952213c1afcd1ad22e9b7a630cfdc619d10f7497978d00a2a9a0d9d95735",
    "src/llama/safetensors.h": "c2c2433fd48e0996c80c0f38b6b40762cf86830ef91b1d5b8fdc69989ce3deca",
    "src/llama/tokenizer.h": "4441ddb15fc56cfb3043cddf4eb717060bfedcd05a404ce18d66a7b3640a58cf",
    "tools/mgwi_pack.py": "1ca1c8260c9e78ef8326ca6f4d37e634182edf3a57cb1ccd832fb362f39a298e",
    "tools/safetensors_to_mgwi.py": "eac9d7f69101ac3aeaeb6dcf656243e58d68193c43e1f2c116b2590d7ab121f6",
    "spikes/mcu-f12-microgpt/models/uniform-f12.mgw":
        "742cbd6d0b750bf3d164a23d97390171e3fe545ee9d87a2b0e843d3d8d1ae9f4",
    "spikes/mcu-f12-microgpt/models/uniform-f12.mgwi":
        "6f2e5e2e97840e65a4c9edc0a70f48cede5e02f15bd2a0dbe8285b5d9bf81e51",
    "spikes/mcu-f12-microgpt/models/uniform-f12-corrupt.mgwi":
        "2058ea8e2383bbf795306ebeb2bde80c903945b837269446d6adb0591bd9c6a4",
    "spikes/safetensors-direct-mgwi/results/2026-07-31-amd-x86_64/gate-result.json":
        "f2c4c20f1ed1c720bf17b598790a70e8d5be753671626509282966ff22bd94ce",
    "spikes/safetensors-direct-mgwi/results/2026-07-31-amd-x86_64/correctness-result.json":
        "ed348b9fb8d82932495cae5476a327aba7d74761f1d5cd77273a398f2f39adfe",
}

TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".h", ".ini", ".json", ".md",
    ".patch", ".py", ".sh", ".tsv", ".txt",
}

FORBIDDEN_TEXT = (
    re.compile(r"swarm-spec|\bSWARM\b", re.IGNORECASE),
    re.compile(r"\b(?:Codex|Claude|Gemini)\b"),
    re.compile(r"/Users/|/home/|/mnt/ramdisk/"),
    re.compile(r"\bdns[89]\b|pi5\.local", re.IGNORECASE),
    re.compile(r"\bssh\s+(?:-[^\s]+\s+)*[^\s]+@", re.IGNORECASE),
    re.compile(r"\b(?:192\.168\.|81\.183\.)"),
    re.compile(r"\bD-[0-9]+\b"),
    re.compile(r"\boperator decision\b|\bnext operator\b", re.IGNORECASE),
    re.compile(r"review[- ]lock", re.IGNORECASE),
)

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"release check: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


@lru_cache(maxsize=1)
def publication_paths() -> tuple[Path, ...]:
    """Return Git-tracked files, or all files in an exported tree."""
    if (ROOT / ".git").exists():
        try:
            completed = subprocess.run(
                [
                    "git", "-C", str(ROOT), "ls-files", "-z", "--cached",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError:
            completed = None
        if completed is not None and completed.returncode == 0:
            paths: list[Path] = []
            for encoded in completed.stdout.split(b"\0"):
                if not encoded:
                    continue
                relative = Path(os.fsdecode(encoded))
                if relative.is_absolute() or ".." in relative.parts:
                    fail(f"Git returned unsafe publication path: {relative}")
                path = ROOT / relative
                if path.exists() or path.is_symlink():
                    paths.append(path)
            return tuple(sorted(paths, key=lambda path: str(path.relative_to(ROOT))))

    paths = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "archive" in relative.parts:
            continue
        paths.append(path)
    return tuple(sorted(paths, key=lambda path: str(path.relative_to(ROOT))))


def check_structure() -> None:
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            fail(f"required regular file missing: {relative}")
    for relative in ABSENT:
        if (ROOT / relative).exists():
            fail(f"internal path present in publication tree: {relative}")
    for path in publication_paths():
        if path.is_symlink():
            fail(f"symlink is not permitted: {path.relative_to(ROOT)}")


def check_hashes() -> None:
    for relative, expected in PINNED_SHA256.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            fail(f"SHA-256 mismatch for {relative}: {actual}")


def check_results() -> None:
    summary = json.loads((ROOT / "results/tinyllama_summary.json").read_text())
    combined = summary["validation"]["combined_decisions"]
    if combined != {
        "matched": 632,
        "evaluated": 632,
        "maximum_requested": 640,
        "prompts": 32,
        "early_eos_prompts": 5,
    }:
        fail("TinyLlama summary does not contain corrected 632/632 accounting")

    gate = json.loads((ROOT / (
        "spikes/safetensors-direct-mgwi/results/2026-07-31-amd-x86_64/"
        "gate-result.json"
    )).read_text())
    if gate.get("status") != "PASS":
        fail("direct conversion gate is not PASS")
    comparison = gate["full_file_comparison"]
    if not comparison["byte_identical"] or comparison["bytes_compared"] != 2_200_116_192:
        fail("direct conversion full-file identity is inconsistent")

    correctness = json.loads((ROOT / (
        "spikes/safetensors-direct-mgwi/results/2026-07-31-amd-x86_64/"
        "correctness-result.json"
    )).read_text())
    combined = correctness["combined"]
    if combined["matched_greedy_decisions"] != 632:
        fail("direct correctness result does not report 632 matched decisions")
    if combined["total_evaluated_greedy_decisions"] != 632:
        fail("direct correctness evaluated-decision count is inconsistent")
    if combined["raw_values"] != 3_264_000:
        fail("direct correctness raw-logit count is inconsistent")


def check_hygiene() -> None:
    for path in publication_paths():
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile":
            continue
        if path == ROOT / "scripts/check_release.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT)
        for pattern in FORBIDDEN_TEXT:
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                fail(f"publication-sensitive text in {relative}:{line}: {match.group(0)!r}")


def check_links() -> None:
    for path in publication_paths():
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                fail(f"Markdown link escapes repository in {path.relative_to(ROOT)}: {target}")
            if not resolved.exists():
                fail(f"broken Markdown link in {path.relative_to(ROOT)}: {target}")


def main() -> int:
    check_structure()
    check_hashes()
    check_results()
    check_hygiene()
    check_links()
    print("RELEASE_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen TinyLlama behavior and raw-logit confirmation gates.

The schedule requests at most 640 greedy decisions. Five confirmation prompts
stop at a matching EOS, so the actual pinned count is 632 evaluated decisions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ORACLE_BYTES = 8_800_406_496
ORACLE_SHA256 = "7e8218d7f79a784f9d1868140fb16c3b9f5fbc45c19fb5c807ddcba5b41e32a8"
ROUNDED_BYTES = 8_800_406_496
ROUNDED_SHA256 = "7ec8e1fd6442c8ad467603460a8cadd3c7d45b3e8fdde94c0cf8b8408fad4b45"
PACKED_BYTES = 2_200_116_192
PACKED_SHA256 = "fa733f5afdec220a91fbaae17ce00bcc20685f25afadd3eec92cfde0af1192c7"
REFERENCE_SHA256 = "4096d4e1b05ef56cef8d5d8c28904cc927350b564b56f080b02136bb22a56114"
TOKENIZER_SHA256 = "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e"
HELDOUT_SHA256 = "6a406b448c3b8544c675a15d3f367eaca3da6b25b6ef4d185fdc8bfb60441aa4"
CONFIRMATION_SHA256 = "e643d59859a655b756434f19b83b4c4ef8c9394fc634e38d3d579cf7af6f8f4e"
RAW_FNV64 = "2a32782dc6e68e12"
RAW_VECTORS = 102
RAW_VALUES = 3_264_000
RAW_BYTES = 26_112_000
MAXIMUM_REQUESTED_DECISIONS = 640
EXPECTED_EVALUATED_DECISIONS = 632
EXPECTED_EARLY_EOS_EMITTED = {
    "confirmation/alphabet": 19,
    "confirmation/capital_japan": 17,
    "confirmation/chemical": 18,
    "confirmation/planet": 17,
    "confirmation/triangle": 16,
}
PROCESS_TIMEOUT_SECONDS = 180
HASH_CHUNK = 8 * 1024 * 1024

TOTAL_PATTERN = re.compile(r"^\s*TOTAL:\s+80/80 tokens match \(100\.0%\)\s*$", re.MULTILINE)
RAW_PATTERN = re.compile(
    r"^RAW_LOGITS_FNV64=([0-9a-f]{16}) vectors=([0-9]+)$",
    re.MULTILINE,
)
GENERATED_PATTERN = re.compile(r"^\s*Generated ([0-9]+) tokens in ", re.MULTILINE)
TOKENS_PATTERN = re.compile(r"^\s*Tokens: \[([0-9, ]+)\]\s*$", re.MULTILINE)


class CorrectnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str


@dataclass(frozen=True)
class Generation:
    tokens: tuple[int, ...]
    emitted_tokens: int
    evaluated_decisions: int


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def checked_regular(
    path: Path,
    label: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise CorrectnessError(f"{label} is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CorrectnessError(f"{label} is not a regular file")
    if expected_size is not None and metadata.st_size != expected_size:
        raise CorrectnessError(
            f"{label} size mismatch: expected {expected_size}, found {metadata.st_size}"
        )
    digest = sha256_path(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CorrectnessError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )
    return {"bytes": metadata.st_size, "sha256": digest}


def parse_prompts(path: Path, expected_sha256: str, expected_count: int) -> list[Prompt]:
    checked_regular(path, path.name, expected_sha256=expected_sha256)
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line:
            continue
        pieces = raw_line.split("\t")
        if len(pieces) != 2 or not pieces[0] or not pieces[1]:
            raise CorrectnessError(f"{path.name}:{line_number}: expected ID<TAB>PROMPT")
        prompt_id, text = pieces
        if prompt_id in seen or not re.fullmatch(r"[a-z0-9_]+", prompt_id):
            raise CorrectnessError(f"{path.name}:{line_number}: invalid/duplicate ID")
        seen.add(prompt_id)
        prompts.append(Prompt(prompt_id, text))
    if len(prompts) != expected_count:
        raise CorrectnessError(
            f"{path.name}: expected {expected_count} prompts, found {len(prompts)}"
        )
    return prompts


def create_result_directory(path: Path) -> Path:
    parent = path.parent.resolve(strict=True)
    if parent.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise CorrectnessError("result parent must not be group/world writable")
    result = parent / path.name
    try:
        result.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise CorrectnessError("refusing to reuse result directory") from exc
    return result


def write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise CorrectnessError(f"short write for {path.name}")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)


def run_model(
    binary: Path,
    model: Path,
    mode_flag: str,
    ref_dir: Path,
    output_dir: Path,
    label: str,
    extra: list[str],
    dump_logits: bool,
) -> tuple[str, Path | None, float]:
    stdout_path = output_dir / f"{label}.stdout"
    stderr_path = output_dir / f"{label}.stderr"
    dump_path = output_dir / f"{label}.logits.bin" if dump_logits else None
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "INTLLM_HASH_LOGITS": "1",
    }
    if dump_path is not None:
        environment["INTLLM_DUMP_LOGITS"] = str(dump_path)
    command = [
        str(binary),
        str(model),
        mode_flag,
        "--ref-dir",
        str(ref_dir),
        *extra,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CorrectnessError(f"{label} exceeded {PROCESS_TIMEOUT_SECONDS}s") from exc
    elapsed = time.monotonic() - started
    write_exclusive(stdout_path, completed.stdout)
    write_exclusive(stderr_path, completed.stderr)
    if completed.returncode != 0:
        raise CorrectnessError(f"{label} exited {completed.returncode}")
    if completed.stderr:
        raise CorrectnessError(f"{label} wrote stderr")
    try:
        stdout = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorrectnessError(f"{label} stdout is not UTF-8") from exc
    return stdout, dump_path, elapsed


def parse_benchmark(
    stdout: str,
    label: str,
    expected_fingerprint: str | None,
) -> dict[str, Any]:
    if len(TOTAL_PATTERN.findall(stdout)) != 1:
        raise CorrectnessError(f"{label}: missing or duplicate 80/80 total")
    raw_matches = RAW_PATTERN.findall(stdout)
    if len(raw_matches) != 1:
        raise CorrectnessError(f"{label}: missing or duplicate raw-logit summary")
    fingerprint, raw_vectors = raw_matches[0]
    if int(raw_vectors) != RAW_VECTORS:
        raise CorrectnessError(
            f"{label}: expected {RAW_VECTORS} raw vectors, found {raw_vectors}"
        )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise CorrectnessError(
            f"{label}: expected raw fingerprint {expected_fingerprint}, "
            f"found {fingerprint}"
        )
    return {
        "matched_decisions": 80,
        "total_decisions": 80,
        "raw_fnv64": fingerprint,
        "raw_vectors": int(raw_vectors),
    }


def parse_generation(stdout: str, label: str) -> Generation:
    generated = GENERATED_PATTERN.findall(stdout)
    tokens = TOKENS_PATTERN.findall(stdout)
    if len(generated) != 1:
        raise CorrectnessError(f"{label}: expected exactly one generation summary")
    emitted = int(generated[0])
    if not 1 <= emitted <= 20:
        raise CorrectnessError(f"{label}: invalid emitted-token count {emitted}")
    if len(tokens) != 1:
        raise CorrectnessError(f"{label}: expected exactly one token vector")
    values = tuple(int(item.strip()) for item in tokens[0].split(","))
    if len(values) < emitted:
        raise CorrectnessError(f"{label}: token vector is shorter than generated suffix")
    # generate() does not append EOS to the returned token vector.  When it
    # emits fewer than the requested 20 tokens, the next greedy decision was
    # the matching EOS that stopped both lanes and must be counted.
    evaluated = emitted if emitted == 20 else emitted + 1
    return Generation(values, emitted, evaluated)


def compare_files(left: Path, right: Path, expected_size: int) -> dict[str, Any]:
    left_info = checked_regular(left, left.name, expected_size=expected_size)
    right_info = checked_regular(right, right.name, expected_size=expected_size)
    compared = 0
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while compared < expected_size:
            amount = min(HASH_CHUNK, expected_size - compared)
            left_chunk = left_stream.read(amount)
            right_chunk = right_stream.read(amount)
            if len(left_chunk) != amount or len(right_chunk) != amount:
                raise CorrectnessError("raw-logit dump truncated during comparison")
            if left_chunk != right_chunk:
                mismatch = next(
                    index
                    for index, (left_byte, right_byte) in enumerate(
                        zip(left_chunk, right_chunk, strict=True)
                    )
                    if left_byte != right_byte
                )
                raise CorrectnessError(
                    f"raw-logit dumps differ at byte {compared + mismatch}"
                )
            compared += amount
    return {
        "bytes_compared": compared,
        "values_compared": compared // 8,
        "rounded_wide_sha256": left_info["sha256"],
        "packed_sha256": right_info["sha256"],
        "byte_identical": True,
    }


def token_hash(tokens: tuple[int, ...]) -> str:
    canonical = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def write_result(path: Path, result: dict[str, Any]) -> None:
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_exclusive(path, payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    runner_sha = sha256_path(runner_path)
    result_dir = create_result_directory(args.output_dir)
    binary = args.binary.resolve(strict=True)
    oracle = args.oracle_mgw.resolve(strict=True)
    rounded = args.rounded_mgw.resolve(strict=True)
    packed = args.packed_mgwi.resolve(strict=True)
    ref_dir = args.ref_dir.resolve(strict=True)

    identities = {
        "binary": checked_regular(binary, "binary"),
        "oracle_mgw": checked_regular(
            oracle,
            "oracle MGW",
            expected_size=ORACLE_BYTES,
            expected_sha256=ORACLE_SHA256,
        ),
        "rounded_mgw": checked_regular(
            rounded,
            "rounded-wide MGW",
            expected_size=ROUNDED_BYTES,
            expected_sha256=ROUNDED_SHA256,
        ),
        "packed_mgwi": checked_regular(
            packed,
            "packed MGWI",
            expected_size=PACKED_BYTES,
            expected_sha256=PACKED_SHA256,
        ),
        "reference_tokens": checked_regular(
            ref_dir / "reference_tokens.txt",
            "reference tokens",
            expected_sha256=REFERENCE_SHA256,
        ),
        "tokenizer_json": checked_regular(
            ref_dir / "tokenizer.json",
            "tokenizer.json",
            expected_sha256=TOKENIZER_SHA256,
        ),
        "heldout_prompts": checked_regular(
            args.heldout_prompts,
            "heldout prompts",
            expected_sha256=HELDOUT_SHA256,
        ),
        "confirmation_prompts": checked_regular(
            args.confirmation_prompts,
            "confirmation prompts",
            expected_sha256=CONFIRMATION_SHA256,
        ),
    }
    heldout = parse_prompts(args.heldout_prompts, HELDOUT_SHA256, 8)
    confirmation = parse_prompts(
        args.confirmation_prompts,
        CONFIRMATION_SHA256,
        20,
    )

    print("DIRECT_MGWI_CORRECTNESS phase=public-oracle", flush=True)
    oracle_stdout, _unused_dump, oracle_elapsed = run_model(
        binary,
        oracle,
        "--native",
        ref_dir,
        result_dir,
        "public-oracle",
        ["--benchmark"],
        False,
    )
    oracle_public = parse_benchmark(oracle_stdout, "public oracle", None)

    print("DIRECT_MGWI_CORRECTNESS phase=public-rounded", flush=True)
    rounded_stdout, rounded_dump, rounded_elapsed = run_model(
        binary,
        rounded,
        "--native",
        ref_dir,
        result_dir,
        "public-rounded",
        ["--benchmark"],
        True,
    )
    rounded_public = parse_benchmark(
        rounded_stdout,
        "public rounded",
        RAW_FNV64,
    )

    print("DIRECT_MGWI_CORRECTNESS phase=public-packed", flush=True)
    packed_stdout, packed_dump, packed_elapsed = run_model(
        binary,
        packed,
        "--native-int16",
        ref_dir,
        result_dir,
        "public-packed",
        ["--benchmark"],
        True,
    )
    packed_public = parse_benchmark(packed_stdout, "public packed", RAW_FNV64)
    if rounded_dump is None or packed_dump is None:
        raise CorrectnessError("internal error: benchmark dumps were not configured")
    raw_comparison = compare_files(rounded_dump, packed_dump, RAW_BYTES)
    if raw_comparison["values_compared"] != RAW_VALUES:
        raise CorrectnessError("raw-logit value count mismatch")

    prompt_results: list[dict[str, Any]] = []
    matched = 80
    for group_name, prompts in (("heldout", heldout), ("confirmation", confirmation)):
        for prompt_index, prompt in enumerate(prompts, 1):
            print(
                f"DIRECT_MGWI_CORRECTNESS phase={group_name} "
                f"prompt={prompt_index}/{len(prompts)} id={prompt.prompt_id}",
                flush=True,
            )
            oracle_stdout, _unused, oracle_seconds = run_model(
                binary,
                oracle,
                "--native",
                ref_dir,
                result_dir,
                f"{group_name}-{prompt.prompt_id}-oracle",
                ["--generate", "--prompt", prompt.text, "--max-new-tokens", "20"],
                False,
            )
            packed_stdout, _unused, packed_seconds = run_model(
                binary,
                packed,
                "--native-int16",
                ref_dir,
                result_dir,
                f"{group_name}-{prompt.prompt_id}-packed",
                ["--generate", "--prompt", prompt.text, "--max-new-tokens", "20"],
                False,
            )
            oracle_tokens = parse_generation(
                oracle_stdout,
                f"{group_name}/{prompt.prompt_id}/oracle",
            )
            packed_tokens = parse_generation(
                packed_stdout,
                f"{group_name}/{prompt.prompt_id}/packed",
            )
            if oracle_tokens.tokens != packed_tokens.tokens:
                raise CorrectnessError(
                    f"token mismatch for {group_name}/{prompt.prompt_id}"
                )
            if oracle_tokens.emitted_tokens != packed_tokens.emitted_tokens:
                raise CorrectnessError(
                    f"stop-length mismatch for {group_name}/{prompt.prompt_id}"
                )
            prompt_key = f"{group_name}/{prompt.prompt_id}"
            expected_emitted = EXPECTED_EARLY_EOS_EMITTED.get(prompt_key, 20)
            if oracle_tokens.emitted_tokens != expected_emitted:
                raise CorrectnessError(
                    f"{prompt_key}: expected {expected_emitted} emitted tokens, "
                    f"found {oracle_tokens.emitted_tokens}"
                )
            matched += oracle_tokens.evaluated_decisions
            prompt_results.append(
                {
                    "group": group_name,
                    "id": prompt.prompt_id,
                    "emitted_tokens": oracle_tokens.emitted_tokens,
                    "matched_greedy_decisions": oracle_tokens.evaluated_decisions,
                    "early_eos": oracle_tokens.emitted_tokens < 20,
                    "token_vector_sha256": token_hash(oracle_tokens.tokens),
                    "oracle_seconds": oracle_seconds,
                    "packed_seconds": packed_seconds,
                }
            )

    if matched != EXPECTED_EVALUATED_DECISIONS or len(prompt_results) != 28:
        raise CorrectnessError(
            "internal error: incomplete or miscounted correctness schedule"
        )
    if sha256_path(runner_path) != runner_sha:
        raise CorrectnessError("runner source changed during execution")
    if sha256_path(binary) != identities["binary"]["sha256"]:
        raise CorrectnessError("binary changed during execution")

    result = {
        "schema": "direct_mgwi_correctness_gate_v1",
        "status": "PASS",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host_class": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "identities": identities,
        "runner_sha256": runner_sha,
        "public_gate": {
            "prompts": 4,
            "oracle": oracle_public,
            "rounded": rounded_public,
            "packed": packed_public,
            "oracle_seconds": oracle_elapsed,
            "rounded_seconds": rounded_elapsed,
            "packed_seconds": packed_elapsed,
        },
        "raw_logit_comparison": raw_comparison,
        "prompt_results": prompt_results,
        "combined": {
            "prompts": 32,
            "matched_greedy_decisions": matched,
            "total_evaluated_greedy_decisions": EXPECTED_EVALUATED_DECISIONS,
            "maximum_requested_decisions": MAXIMUM_REQUESTED_DECISIONS,
            "early_eos_prompts": len(EXPECTED_EARLY_EOS_EMITTED),
            "raw_fnv64": RAW_FNV64,
            "raw_vectors": RAW_VECTORS,
            "raw_values": RAW_VALUES,
        },
    }
    write_result(result_dir / "correctness-result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--oracle-mgw", type=Path, required=True)
    parser.add_argument("--rounded-mgw", type=Path, required=True)
    parser.add_argument("--packed-mgwi", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--heldout-prompts", type=Path, required=True)
    parser.add_argument("--confirmation-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args())
    combined = result["combined"]
    print(
        "DIRECT_MGWI_CORRECTNESS PASS "
        f"prompts={combined['prompts']} "
        f"decisions={combined['matched_greedy_decisions']}/"
        f"{combined['total_evaluated_greedy_decisions']} "
        f"maximum_requested={combined['maximum_requested_decisions']} "
        f"raw_fnv64={combined['raw_fnv64']} "
        f"raw_values={combined['raw_values']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CorrectnessError, OSError, ValueError) as exc:
        print(f"DIRECT_MGWI_CORRECTNESS FAIL: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

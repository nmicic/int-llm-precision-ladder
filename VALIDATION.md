# Validation

The publication tree has one self-contained, non-network entry point:

```sh
sh scripts/validate.sh local
```

## Modes

```text
quick   structure, hashes, corrected result accounting, publication hygiene,
        Markdown links, Python syntax, shell syntax, and git diff checks
local   quick plus native builds, unit tests, committed-model inference,
        portable/native MicroGPT fixture, and host GPU arithmetic tests
status  external-suite boundary only; no tests are run
```

The local command does not use SSH, the network, `sudo`, a GPU, or a serial
uploader. It does not download or execute TinyLlama and does not flash a
physical board.

## What local validation proves

- the committed MicroGPT model and core source hashes match their pins;
- MGW rounding and MGWI packing pass malformed-input and round-trip tests;
- direct safetensors conversion helpers pass their self-contained format and
  rounding tests;
- the C F16/BF16/F32 decoder handles Q16.48 endpoints, finite overflow, and
  non-finite inputs without signed-overflow or conversion undefined behavior;
- caller-supplied `CFLAGS` cannot remove the include paths, feature defines,
  wrap semantics, or C language level required by the integer runtimes;
- the held-out gate parser handles actual generated lengths and early EOS;
- the uniform-F12 MicroGPT packed and rounded-wide host lanes have identical
  samples and raw-logit hashes under native and forced-portable wide math;
- the one-code corrupted model changes the raw-logit hash and is rejected;
- the host-side CUDA arithmetic helpers pass without requiring CUDA hardware;
- public documentation has no dangling relative links or known internal/path
  markers;
- the retained direct-conversion and correctness JSON records agree with the
  documented byte counts and corrected 632/632 decision count.

The committed MicroGPT fixture generator runs in read-only check mode during
`make test`, so validation on a different compiler must not rewrite the
tracked manifest. Deliberate fixture regeneration requires
`python3 spikes/mcu-f12-microgpt/prepare.py --refresh`.

## External suites

The following tests are deliberately not automatic:

- direct TinyLlama conversion and the 632-decision gate require the pinned
  2.2 GB safetensors file, 2.2 GB MGWI, and 8.8 GB MGW artifacts;
- physical MCU results require their respective boards and upload tools;
- CUDA timing requires compatible NVIDIA hardware and the recorded workload
  boundaries.

Removing documentation, internal records, raw logs, or machine-local build
products does not require repeating those external runs. A change to model
bytes, arithmetic, generated model headers, firmware logic, a compiler/profile
pin, a runtime lane, or a claim-bearing external validator does require the
affected external test to be rerun.

Host validation targets POSIX macOS and Linux. The large-file Python tools use
`os.pread`/`os.pwrite`. The C tokenizer fallback and `--dump-reference`
generator accept trusted local model paths; embedded quotes or newlines are
outside the supported path boundary.

## Publication history

The working repository previously contained non-public process records and raw
machine captures. Those are preserved in an ignored local `archive/` but must
not be published. Release from a fresh history containing only the current
tracked publication tree; do not expose the earlier working history.

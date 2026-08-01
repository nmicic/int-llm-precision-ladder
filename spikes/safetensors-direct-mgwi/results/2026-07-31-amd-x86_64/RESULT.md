# Direct safetensors-to-MGWI result

Status: **PASS** for the pinned conversion, full-file identity, unpacked-value,
behavioral, and packed-versus-wide execution gates.

Run date: 2026-07-31

Host class: Linux x86-64

Performance claim: none; conversion and inference durations in the manifests
are descriptive only

## Representation

The converter maps the pinned unsharded BF16/F16 TinyLlama checkpoint directly
to the existing mixed candidate:

```text
default F11
self-attention projections F12
MLP projections            F12
physical payload            little-endian signed int16
```

`F11` and `F12` name fractional-grid precision. They do not mean 11-bit or
12-bit physical fields: every weight code occupies one signed 16-bit integer.
Activations, KV state, nonlinear functions, and accumulators remain
Q16.48/wider.

## Full conversion identity

| Check | Result |
| --- | --- |
| Source safetensors | 2,200,119,864 bytes; `6e6001da...f14933` |
| Direct MGWI | 2,200,116,192 bytes; `fa733f5a...192c7` |
| Direct vs retained MGWI | all 2,200,116,192 bytes identical |
| Logical weight codes | all 1,100,048,384 signed-int16 values identical |
| Unpacked direct vs rounded-wide MGW | all values identical; `7ec8e1fd...4b45` |

The direct conversion took 91.532 seconds and the complete gate 101.088
seconds on this host. These are not benchmark results.

## Fresh correctness confirmation

- The original Q16.48 lane still passed its public 80/80 gate.
- The rounded-wide and direct-packed lanes each passed that same 80/80 gate.
- All 32 prompts matched on **632/632 actually evaluated greedy decisions**.
  The schedule requested at most 640; five confirmation prompts selected EOS
  early in both lanes.
- The rounded-wide and packed lanes produced the same raw fingerprint
  `2a32782dc6e68e12` across 102 vectors / 3,264,000 int64 logits.
- Their two 26,112,000-byte raw-logit dumps were byte-identical, each with
  SHA-256 `d2241dee...e8050`.

The original oracle's raw fingerprint is `ad2607d6614dc6f7`. That is expected:
the rounded model is behaviorally equal on the checked decisions but not
numerically identical to the original Q16.48 model. Raw identity is claimed
only between the wide and packed forms of the same rounded candidate.

## Verifier audit trail

Two fail-closed attempts preceded the retained PASS:

1. The first runner incorrectly expected the rounded candidate's raw hash from
   the original Q16.48 oracle. It stopped, exposing a comparison-role error.
2. The second correctly separated those roles but required every custom prompt
   to emit 20 tokens. It stopped at an existing early-EOS prompt and exposed
   the old shell gate's hard-coded `20/20` report.
3. The retained runner pins the five early-EOS counts, counts the EOS choice as
   an evaluated decision, and passed at 632/632.

The failed attempts are part of the audit history; they are not presented as
successful runs.

## Pinned build and evidence

The fresh release binary was 194,288 bytes with SHA-256
`cb86f663...a5954`. It was built with GCC 13.3 release flags; the compiler
binary SHA-256 was
`1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26`.
Its relevant source identities were:

```text
llama_int.c   73fc5789e60e10f10ed351a790f62aa00ebe3f2e8faea87edfa65bc405b91166
fp_math.h     38520b39b86f5fe90f7d92cd3a3026877e16370f15a85c9b5f22079e291abf6c
safetensors.h 870755bd6ed8a391b813bd7f4b94084094e10c0cc709d88a18ad3cc4ed92faae
tokenizer.h   4441ddb15fc56cfb3043cddf4eb717060bfedcd05a404ce18d66a7b3640a58cf
```

Machine-readable evidence:

- [`gate-result.json`](gate-result.json), SHA-256
  `f2c4c20f1ed1c720bf17b598790a70e8d5be753671626509282966ff22bd94ce`
- [`correctness-result.json`](correctness-result.json), SHA-256
  `ed348b9fb8d82932495cae5476a327aba7d74761f1d5cd77273a398f2f39adfe`

## Limits

- This converter is bounded to the pinned single-file BF16/F16 checkpoint and
  exact mixed plan; it is not a generic safetensors converter.
- MGWI remains an experimental trusted-input format, not a hostile-file
  security contract.
- The result proves exact reproduction of the existing compact candidate; it
  does not show that the rounded model is numerically identical to Q16.48.
- No MCU, GPU, cross-host performance, or floating-point performance claim is
  made by this gate.

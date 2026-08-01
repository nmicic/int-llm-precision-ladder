# Provenance

## Source base

This repository began from public `int-llm` commit:

```text
1b80e024c5fadea955e3892578a36dbc80a8a0b5f
```

The committed small-model oracle retains its original identity. The
fixed-point library entered this work with the following inherited identity;
its current publication hash differs only because stale provenance and test
target comments were corrected:

```text
466cfe9dba7b888cdaa23dedf4b10351826795793448c8e95dcb0f7a61ed33eb  model.mgw
38520b39b86f5fe90f7d92cd3a3026877e16370f15a85c9b5f22079e291abf6c  src/fp_math.h
```

`src/microgpt_int.c` adds opt-in precision tracing, packed-F12 inference, and
fixture hooks around the inherited model implementation. These features are
compile-time or runtime gated and do not alter the normal training path.

## Publication source identities

```text
b25fc25ec2c7f0f6fd9511412f53d46473cc6a89e26d14af4d32b9ba8a0021d4  src/microgpt_int.c
21d6952213c1afcd1ad22e9b7a630cfdc619d10f7497978d00a2a9a0d9d95735  src/fp_math.h
5908bf5bcd224ac8528e21fb113ff466ea497643f6248e67ce343545c01a560c  src/llama/llama_int.c
21d6952213c1afcd1ad22e9b7a630cfdc619d10f7497978d00a2a9a0d9d95735  src/llama/fp_math.h
c2c2433fd48e0996c80c0f38b6b40762cf86830ef91b1d5b8fdc69989ce3deca  src/llama/safetensors.h
4441ddb15fc56cfb3043cddf4eb717060bfedcd05a404ce18d66a7b3640a58cf  src/llama/tokenizer.h
1ca1c8260c9e78ef8326ca6f4d37e634182edf3a57cb1ccd832fb362f39a298e  tools/mgwi_pack.py
eac9d7f69101ac3aeaeb6dcf656243e58d68193c43e1f2c116b2590d7ab121f6  tools/safetensors_to_mgwi.py
```

The tokenizer line above intentionally records the inherited file identity;
the exact checksum is also checked during validation.

The current safetensors decoder uses range-checked unsigned scaling for F16,
BF16, and F32. Finite values outside Q16.48 saturate at the signed endpoints;
the inherited non-finite policy remains ±32000. The direct MGWI converter is
deliberately stricter and rejects out-of-range source weights. These boundary
rules do not change any admitted TinyLlama weight used by the retained gates.

## TinyLlama runtime lineage

The isolated TinyLlama runtime began from these inherited files:

```text
b75a4ccbb20a96ed7d42e5f28c57ecf441308a9d8ea115ee37940ed4876d75bb  llama_int.c
38520b39b86f5fe90f7d92cd3a3026877e16370f15a85c9b5f22079e291abf6c  fp_math.h
63fdc6caa907d5e00e38912da49fe543aa609446be59186ef3095ea39083c469  safetensors.h
4441ddb15fc56cfb3043cddf4eb717060bfedcd05a404ce18d66a7b3640a58cf  tokenizer.h
```

The runtime then gained MGWI loading, tagged signed-int16 kernels, raw-logit
gates, descriptor-bound test inputs, and benchmark-boundary counters. The
direct conversion/correctness capture used `llama_int.c` SHA-256
`73fc5789e60e10f10ed351a790f62aa00ebe3f2e8faea87edfa65bc405b91166`.
The current `5908...60c` source adds generation-boundary profiling and a
token-only benchmark mode; it does not change model arithmetic.

The capture-time `safetensors.h` SHA-256 was `870755...92faae`; the historical
result file retains that value. Its direct converter SHA-256 was
`61b644...38ec1f`; the current converter rejects non-finite and extreme-shift
inputs explicitly. The current changes only harden behavior outside the
admitted TinyLlama weight domain.

## Model lineage

```text
7e8218d7f79a784f9d1868140fb16c3b9f5fbc45c19fb5c807ddcba5b41e32a8  exact Q16.48 TinyLlama MGW
7ec8e1fd6442c8ad467603460a8cadd3c7d45b3e8fdde94c0cf8b8408fad4b45  rounded mixed-F11/F12 MGW
fa733f5afdec220a91fbaae17ce00bcc20685f25afadd3eec92cfde0af1192c7  packed signed-int16 MGWI
```

The direct safetensors converter recreated the packed MGWI byte-for-byte. Its
unpacked values recreated the rounded-wide MGW byte-for-byte. TinyLlama model
artifacts are external because of their size and are not included here.

## Publication-only sanitization

Publication preparation removed internal process records and machine-specific
captures. Executable arithmetic logic and model bytes were not changed. Two
comments in `src/microgpt_int.c` and three include comments in board wrappers
were made neutral; the regenerated native and forced-portable MicroGPT host
fixtures remained identical and passed the positive and corruption gates.
No physical-board claim was expanded or rerun because the only firmware-source
edits were comments, while model bytes, generated headers, compiler/profile
pins, and executable logic were unchanged. Pre-publication host-loader
hardening affected only out-of-range floating inputs and is covered by focused
host tests; it does not execute in the retained MCU lanes.

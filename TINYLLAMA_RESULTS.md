# TinyLlama mixed-F11/F12 and packed-integer results

Run dates: 2026-07-30--31

Primary host: AMD Ryzen 7 7700, Linux x86-64

GPU: not used

## Headline

A mixed-F11/F12 TinyLlama-1.1B candidate matched the original Q16.48 integer
oracle on **632 of 632 actually evaluated greedy decisions across 32 prompts**.
The schedule requested at most 640 decisions; five confirmation prompts
reached matching EOS early.

Every rounded weight fits signed int16. Executing those codes directly:

- preserved all 632 checked greedy decisions;
- reproduced the rounded-wide path's raw-logit bytes across 102 forward calls
  and 3,264,000 int64 logits;
- reduced the weight file from 8,800,406,496 to 2,200,116,192 bytes;
- reduced median peak RSS by 73.303% against the rounded-wide integer path;
- reduced median generation time by 23.209% on the tested AMD host.

`F12` means a grid with 12 fractional bits, not a 12-bit container. Each F11
or F12 code is stored in a signed 16-bit field. Activations, KV cache,
nonlinear functions, and other state remain Q16.48; matrix products retain
signed-128 accumulation.

## Precision boundary

The source oracle is the Q16.48 TinyLlama MGW:

```text
bytes    8,800,406,496
sha256  7e8218d7f79a784f9d1868140fb16c3b9f5fbc45c19fb5c807ddcba5b41e32a8
weights 1,100,048,384 across 201 tensors
```

The tested boundary was:

| Candidate | Four-prompt gate | Eight-prompt gate | Result |
| --- | ---: | ---: | --- |
| Uniform F12 | 80/80 | 160/160 | conservative survivor |
| Uniform F11 | 80/80 | first held-out prompt diverges | rejected |
| Uniform F10 | first gate diverges | not promoted | rejected |
| Mixed F11/F12 | 80/80 | 160/160 | promoted |

The promoted plan is:

```text
default F11
model.layers.*.self_attn.*_proj.weight = F12
model.layers.*.mlp.*_proj.weight       = F12
```

This puts 154 projection tensors / 968,884,224 weights (88.0765%) at F12 and
47 tensors / 131,164,160 weights (11.9235%) at F11. The F11 group contains
embeddings, normalization tensors, and `lm_head`.

Rounded-wide candidate:

```text
sha256 7ec8e1fd6442c8ad467603460a8cadd3c7d45b3e8fdde94c0cf8b8408fad4b45
```

This candidate is behaviorally exact only on the stated greedy gates. Hidden
states and logits drift from the original Q16.48 model. The original oracle's
raw fingerprint is `ad2607d6614dc6f7`; the candidate's is different as
expected.

## Corrected gate accounting

A separately frozen 20-prompt confirmation set added 392 evaluated decisions.
The complete accounting is:

```text
four-prompt gate    4 prompts   80/80
development gate    8 prompts  160/160
confirmation       20 prompts  392/392
total              32 prompts  632/632  (640 maximum requested)
```

The earlier `640/640` wording counted requested output slots rather than
decisions actually evaluated before EOS. This was a reporting defect, not a
model regression. The corrected runner parses each generated length and
counts the matching EOS choice.

Confirmation-prompt manifest SHA-256:

```text
e643d59859a655b756434f19b83b4c4ef8c9394fc634e38d3d579cf7af6f8f4e
```

## Direct safetensors-to-MGWI conversion

The pinned direct converter reads the original single-file TinyLlama
safetensors checkpoint and writes the mixed signed-int16 payload without first
materializing an 8.8 GB rounded MGW.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Source safetensors | 2,200,119,864 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |
| Direct MGWI | 2,200,116,192 | `fa733f5afdec220a91fbaae17ce00bcc20685f25afadd3eec92cfde0af1192c7` |

The direct output was compared with the earlier reference pipeline:

- all 2,200,116,192 packed bytes were identical;
- all 1,100,048,384 signed-int16 codes were identical;
- unpacking recreated the 8.8 GB rounded-wide MGW byte-for-byte.

This closes the practical `safetensors -> MGWI` path for the pinned checkpoint
and precision plan. It is not a generic model converter claim. Detailed
machine-readable records are under
`spikes/safetensors-direct-mgwi/results/2026-07-31-amd-x86_64/`.

## Packed format and arithmetic identity

MGWI stores each weight as signed int16 plus a per-tensor F11/F12 tag. The
packer refuses off-grid values and codes outside the signed-int16 range.

For stored code `s` at fractional precision `b`:

```text
weight_Q48 = s * 2^(48-b)

rounded-wide: sum(input_Q48 * (s << (48-b))) >> 48
packed:       sum(input_Q48 * s)             >> b
```

The two expressions preserve the same arithmetic truncation. Across the full
checked schedule, both paths produced:

```text
RAW_LOGITS_FNV64=2a32782dc6e68e12 vectors=102
```

Their 26,112,000-byte raw-logit dumps were byte-identical, covering 3,264,000
int64 values.

## AMD packed versus rounded-wide integer execution

The retained host comparison used the same rounded model and integer runtime;
only physical weight representation and the corresponding load/kernel route
differed.

| Lane | Generation median | Peak RSS median |
| --- | ---: | ---: |
| rounded-wide int64 MGW | 49.0083 s | 8,469,436,416 B |
| direct signed-int16 MGWI | 37.6339 s | 2,261,084,160 B |

Packed reduced generation time by **23.209%** and peak RSS by **73.303%**.
Both counterbalanced timing cycles cleared the predefined 5% threshold, and
the lanes matched all checked raw logits.

This is a one-host representation-plus-execution result. It does not show
that integer arithmetic itself is faster and should not be generalized across
hosts. Memory capacity is the more transferable signal.

## Separate practical FP32 control

A separate comparison used one-thread FP32 PyTorch/Transformers on the same
BF16 source checkpoint and the same frozen 80-decision schedule. Both FP32 and
packed lanes qualified at 80/80. A BF16-parameter lane reached 78/80 and was
excluded from timing.

| Lane | Generation median | Peak RSS median |
| --- | ---: | ---: |
| FP32 PyTorch/Transformers | 11.9678 s | 7,306,727,424 B |
| packed MGWI C runtime | 37.9248 s | 2,261,082,112 B |

The packed runtime took **3.168899x** as long but used about **69.1% less**
peak RSS. This is a complete-runtime comparison, not a matched arithmetic
benchmark: PyTorch/MKLDNN batches prefill and uses optimized floating-point
kernels, while the C path evaluates one logit row per forward call.

The result supports compact storage and lower memory use. It does not support
an x86 speed claim for the present packed C runtime.

## Signed-width census

An exact tensor-by-tensor census of the packed model found:

| Required signed width | Tensors | Elements |
| ---: | ---: | ---: |
| 9 | 1 | 524,288 |
| 10 | 21 | 73,410,560 |
| 11 | 19 | 102,258,688 |
| 12 | 100 | 477,671,424 |
| 13 | 55 | 418,394,112 |
| 14 | 4 | 27,265,024 |
| 15 | 1 | 524,288 |

Ideal per-tensor bit packing would require about 1,678,072,544 bytes, but
arbitrary-width unpacking may erase the execution benefit. No tensor fits
signed int8 at the current grids; int8 would require more numerical change.

## Reproduce

Build the runtime and local unit tests:

```sh
make test
```

Create the rounded-wide candidate:

```sh
./mgw_round --default-bits 11 \
  --set 'model.layers.*.self_attn.*_proj.weight=12' \
  --set 'model.layers.*.mlp.*_proj.weight=12' \
  exact.mgw rounded.mgw
```

Pack and run it:

```sh
python3 tools/mgwi_pack.py pack --default-bits 11 \
  --set 'model.layers.*.self_attn.*_proj.weight=12' \
  --set 'model.layers.*.mlp.*_proj.weight=12' \
  rounded.mgw rounded.mgwi

./llama_mgwi rounded.mgwi --native-int16 \
  --ref-dir /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --benchmark --profile
```

## Limits

- The behavioral result covers short greedy gates, not arbitrary prompts,
  sampling, or broad quality parity.
- Performance is from one admitted AMD host; it is not a cross-host claim.
- MGWI is experimental and trusted-input-only, not a hardened file format.
- The compact path improves storage and memory behavior; it says nothing by
  itself about CUDA or floating-point arithmetic.
- Q16.48 remains the correctness oracle. The mixed candidate is a lower-
  precision result checked against it, not a replacement reference.

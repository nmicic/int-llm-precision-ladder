# P13 rotated-weight CUDA transferability result

- Capture time: 2026-07-31T03:01:48Z--03:02:08Z
- Target: NVIDIA GeForce RTX 5090, compute capability 12.0
- Toolchain: CUDA 13.3 / nvcc 13.3.73, driver 610.43.02
- Reviewed source commit: `ba3ea872c38db798719c5bbde5b648fd202d0871`
- Scope: synthetic M=1 projection GEMV; one retained observation is a
  12-launch sequence over 12 distinct resident weight copies

## Outcome

**VALID synthetic evidence; frozen criteria PASS; GPU work PAUSED and the
exploration returns to CPU/edge.**

Storing the same signed F12 weight codes in int16 rather than int32 reduced
the median 12-GEMV sequence time by **41.2--42.1%** across both tested shapes
and all three fresh processes. All 72 paired observations passed; their
individual reductions ranged from 40.26% to 43.16%. Every packed and wide
sample stayed inside the predeclared +/-10% lane-median envelope.

Every checked integer output was exact. The FP32 contextual-control outputs
were finite and inside their predeclared bounds. Build identity, working-set
size, pressure traversal, output completeness, process freshness, SASS path,
sanitizer, qualification, and evidence-integrity gates all passed.

This is a **representation-plus-execution** result. It shows that the packed
storage advantage survives a longer unit that rotates through distinct
addresses larger than L2. It does not show that integer arithmetic itself is
faster, and it is not a real-weight, layer, token, model, or runtime result.

The frozen protocol would permit a separate proposal for one real
rounded-weight tensor. GPU work was paused after this capture. No full-model
claim follows from this result.

## Compared lanes

The integer lanes used:

- identical canonical signed F12 weight codes;
- int32 activations;
- signed-int64 accumulation;
- the same mathematical-floor epilogue;
- one warp per output row and identical launch geometry.

Only resident weight storage differed:

- `WIDE_INT32_CONTROL`: one signed code per int32;
- `PACKED_INT16_CANDIDATE`: the same code stored in int16 and sign-extended
  when loaded.

The FP32 lane was a four-byte CUDA-core contextual control. No lane used
cuBLAS, WMMA, tensor cores, or a matrix instruction.

For each lane and shape, the retained unit timed 12 GEMVs over 12 distinct
weight copies. The packed image was 276,824,064 bytes (2.75x reported L2);
the wide and FP32 images were each 553,648,128 bytes (5.50x L2).

## Timing

Sequence medians are the decisive values. Per-launch numbers are sequence
medians divided by 12 and are only a human-readable amortized view.

| Shape | Process | Wide sequence us | Packed sequence us | Reduction | Wide/launch us | Packed/launch us |
|---|---:|---:|---:|---:|---:|---:|
| gate/up | 0 | 367.536 | 215.632 | 41.33% | 30.628 | 17.969 |
| gate/up | 1 | 364.928 | 214.512 | 41.22% | 30.411 | 17.876 |
| gate/up | 2 | 365.328 | 214.720 | 41.23% | 30.444 | 17.893 |
| down | 0 | 386.272 | 223.936 | 42.03% | 32.189 | 18.661 |
| down | 1 | 383.216 | 222.176 | 42.02% | 31.935 | 18.515 |
| down | 2 | 384.368 | 222.448 | 42.13% | 32.031 | 18.537 |

Both counterbalanced cycle medians passed in every process. Their reductions
ranged from 41.02% to 42.21%.

The FP32 sequence medians were 365.088--367.344 us for gate/up and
391.392--393.840 us for down. They are context for this CUDA-core kernel, not
a vendor-library performance ceiling.

## What this resolves

The earlier single-launch P13 candidate showed 35.2--38.9% reductions in the
two pressure-labelled large shapes but failed its complete gate because seven
samples in a separate `repeat_hot` regime exceeded a +/-20% stability
envelope.

This follow-up did not alter or rerun that gate. It asked a new question with
a longer measurement unit, distinct resident weight addresses, stricter
+/-10% stability, and complete output verification after every sequence. It
passed. The valid combined reading is:

1. the old single-launch candidate remains `STOP/PARK`;
2. the narrower resident representation has a repeatable bandwidth-sensitive
   synthetic signal;
3. the signal transfers to the separately frozen rotated sequence;
4. model-level usefulness remains untested.

## Correctness and path evidence

- lossless int16 storage/decode identity passed;
- signed-128 host oracle and int64 overflow admission passed;
- extrema, odd-tail, signed-floor, cancellation, poison, partial-write,
  skipped-slice, and zero-extension controls passed;
- all 12 output slices were checked after each of 72 retained observations in
  each process;
- every pressure traversal advanced and verified the complete 256 MiB
  footprint outside the timed event;
- all six lane-order permutations ran in each of two cycles per shape;
- production qualification ran without retained timing records;
- Compute Sanitizer reported zero errors;
- normalized packed/wide integer opcode structure matched, both integer
  kernels used 32 registers and no local/stack/shared allocation, and the
  complete SASS contained no tensor opcode;
- three distinct runner PIDs, run IDs, and log hashes were bound to strict
  quiescent GPU snapshots;
- the GPU was restored from `Exclusive Process` to `Default`, with no compute
  process left behind. The restore command succeeded during the wrapper, and
  a separate read-only post-capture query recorded `Default` and
  `compute_processes=none`.

## Evidence identity

| Artifact | SHA-256 |
|---|---|
| rotated benchmark source | `44406a4dd9832501482280576f21e7f1d970161601868441c664c52f0ec1c475` |
| frozen protocol | `ba40da8b28efcb5a7a01583f9a3f757648305faf87330253e327639357bd47f0` |
| executable | `fbe5b664480f318a357d10b68aaff61062db43b37f9f023735e849a361ba05be` |
| SASS | `09a5de3a74dccea8baf6b95d6959ca98e8c8dfd3e8366a98cb4b0b91451f7dbf` |
| run manifest | `319b24ee5193e160a122a9a7decb9b68ea7185fdfd5b30558baee20ac076b69f` |
| process 0 log | `1fc178021686ad354fa63cb143f3d0931929edfd3c78b22a0d3b08393bf6b267` |
| process 1 log | `2be5b0ea03a516e825caacd719afa4f0a4b7e49357baad129c122d6ab0929a3b` |
| process 2 log | `c17a4b6b8b20eaa9abc5efeaaf8a306c3f6830a1644ce82d52118a9587f3d7f2` |
| validator result | `eb23c76132f39e93a8b3ff222c43420191a6caf7a1d91061a86448aff4c2d550` |
| supplemental post-restore query | `7d6da672e90f8a9fc39fcf41c976396973396c973f294c15d307d86b9e7c9ae1` |

Machine-local build artifacts, SASS, raw logs, and GPU-mode records are not
part of the public tree. The exact rotated benchmark source is retained under
`spikes/gpu-p13-rotated/` and matches the hash above.

## Disposition

Do not rerun or tune this synthetic harness. It answered its bounded question.

If GPU work is reopened, the next defensible step is one real rounded-weight
tensor with boundary-inclusive costs. This result does not justify jumping
directly to a layer, token loop, or full runtime.

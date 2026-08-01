# Precision-ladder tools

Run these commands from the repository root. `make all` builds the MicroGPT
trace binary and the streaming MGW rounder; `make llama-mgwi` builds the
TinyLlama runtime with native MGWI loading.

The host tools target POSIX macOS and Linux. The large-file Python converters
use `os.pread`/`os.pwrite`; native Windows requires a POSIX environment or a
small I/O portability layer.

| Tool | Purpose |
| --- | --- |
| `mgw_precision.py` | Inspect a MicroGPT MGW, create a uniformly rounded candidate, or verify that a candidate follows the requested fractional grid. |
| `mgw_round.c` | Stream a large MGW into mixed per-tensor fractional grids without loading the whole file; `--analyze-only` reports the result without writing it. |
| `mgwi_pack.py` | Pack an already rounded MGW into tagged signed-int16 MGWI, unpack MGWI into canonical wide MGW, inspect tags, or report ideal storage widths. |
| `safetensors_to_mgwi.py` | Convert the pinned single-file BF16/F16 Llama checkpoint directly to MGWI without materializing an intermediate wide MGW. |
| `run_ladder.py` | Sweep uniform MicroGPT weight precision and record sample, trace, probability, and top-1 comparisons in CSV and JSON. |
| `run_mixed_ladder.py` | Sweep separate MicroGPT body/head precision combinations. |
| `run_tensor_sensitivity.py` | Run MicroGPT tensor-group `only` and `rescue` probes around the precision boundary. |
| `run_heldout_gate.sh` | Compare exact and candidate wide TinyLlama MGWs over a TSV prompt set, including matching early-EOS accounting. |

## MGW to MGWI and back

First round the wide Q16.48 MGW onto the intended per-tensor grids:

```sh
make mgw_round

./mgw_round --default-bits 11 \
  --set 'model.layers.*.self_attn.*_proj.weight=12' \
  --set 'model.layers.*.mlp.*_proj.weight=12' \
  exact.mgw rounded.mgw
```

Then pack and unpack it:

```sh
python3 tools/mgwi_pack.py pack --default-bits 11 \
  --set 'model.layers.*.self_attn.*_proj.weight=12' \
  --set 'model.layers.*.mlp.*_proj.weight=12' \
  rounded.mgw rounded.mgwi

python3 tools/mgwi_pack.py unpack rounded.mgwi restored.mgw
cmp rounded.mgw restored.mgw
```

The packer does not silently round its input. It rejects a selected tensor if
any value is off the declared grid or its scaled code does not fit signed
int16. `--raw GLOB` can retain a matching tensor as int64 when exact raw
storage is required. Output paths must not already exist.

The tested round trip is:

```text
rounded.mgw -> rounded.mgwi -> restored.mgw
                                byte-identical to rounded.mgw
```

It does not recover low bits discarded when converting `exact.mgw` to
`rounded.mgw`. Likewise, unpacking a directly generated MGWI
recreates the corresponding rounded-wide MGW, not the original unrounded
Q16.48 oracle.

`mgwi_pack.py` also provides:

```sh
python3 tools/mgwi_pack.py inspect rounded.mgwi
python3 tools/mgwi_pack.py analyze-widths rounded.mgwi
```

## Direct safetensors conversion

The direct converter requires the expected source, configuration, and output
hashes deliberately. It is an evidence path for a pinned checkpoint and
precision plan, not a general model-conversion frontend:

```sh
python3 tools/safetensors_to_mgwi.py \
  --default-bits 11 \
  --set 'model.layers.*.self_attn.*_proj.weight=12' \
  --set 'model.layers.*.mlp.*_proj.weight=12' \
  --expect-source-sha256 SOURCE_SHA256 \
  --expect-config-sha256 CONFIG_SHA256 \
  --expect-output-sha256 OUTPUT_SHA256 \
  MODEL_DIR direct.mgwi
```

Non-finite BF16/F16 values and values whose Q16.48 magnitude cannot be handled
safely are rejected before a final output is installed.

The destination parent must already exist and must not be group- or
world-writable. On systems using a group-writable default umask, create a
private output directory first, for example `mkdir -m 700 conversion-output`.

Use `mgwi_pack.py unpack` when a canonical rounded-wide MGW is needed from
that output.

## MicroGPT exploration

The common recorded sweeps have Make targets:

```sh
make inspect
make sweep
make sensitivity
make mixed
```

For one candidate rather than a sweep:

```sh
python3 tools/mgw_precision.py inspect model.mgw
python3 tools/mgw_precision.py quantize \
  --input model.mgw --output candidate.mgw --fraction-bits 12
python3 tools/mgw_precision.py verify \
  --input model.mgw --candidate candidate.mgw --fraction-bits 12
```

## TinyLlama held-out gate

`run_heldout_gate.sh` operates on two wide MGW files and a two-column
`ID<TAB>PROMPT` file:

```sh
sh tools/run_heldout_gate.sh \
  LLAMA_INT EXACT.mgw CANDIDATE.mgw MODEL_DIR PROMPTS.tsv RESULT_DIR LABEL
```

It is separate from the direct MGWI correctness runner under
`spikes/safetensors-direct-mgwi/`.

## Validation and trust boundary

```sh
make test
python3 scripts/check_release.py
```

`spikes/mcu-f12-microgpt/prepare.py` verifies the committed generated fixture
without modifying it. Use `--refresh` only when deliberately regenerating
those files; review and commit the resulting changes.

MGWI remains an experimental, trusted-input format. These tools perform
structural and range checks, but the runtime loader is not presented as a
hardened parser for hostile files. The C runtime's fallback tokenizer command
and `--dump-reference` script generator also expect a trusted local model path
without embedded quotes or newlines.

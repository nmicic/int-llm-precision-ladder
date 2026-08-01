#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python3}

usage() {
    cat <<'EOF'
usage: sh scripts/validate.sh MODE

Modes:
  quick   Check release structure, pinned hashes, result consistency, links,
          source syntax, and publication hygiene.
  local   Run quick plus every self-contained native/unit/host-fixture test.
  status  List the external suites that this command deliberately does not run.

The command does not use the network, SSH, sudo, a GPU, or a serial uploader.
EOF
}

run_quick() {
    "$PYTHON" -I -B scripts/check_release.py
    pycache_root=${TMPDIR:-/tmp}/int-llm-precision-release-pyc
    "$PYTHON" -I -B -X "pycache_prefix=$pycache_root" -m py_compile \
        tools/mgw_precision.py \
        tools/mgwi_pack.py \
        tools/safetensors_to_mgwi.py \
        tools/run_ladder.py \
        tools/run_mixed_ladder.py \
        tools/run_tensor_sensitivity.py \
        spikes/safetensors-direct-mgwi/run_gate.py \
        spikes/safetensors-direct-mgwi/run_correctness_gate.py \
        spikes/mcu-f12-microgpt/prepare.py
    sh -n tools/run_heldout_gate.sh
    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git diff --check
    fi
}

run_local() {
    run_quick
    make clean
    make CFLAGS=-O2 all llama-mgwi safetensors_conversion_test
    make clean
    make test
    printf '%s\n' 'LOCAL_VALIDATION=PASS external_suites=NOT_RUN'
}

show_status() {
    cat <<'EOF'
LOCAL            sh scripts/validate.sh local
TINYLLAMA        external 2.2/8.8 GB artifacts required; not run automatically
LIVE MCU         source/profiles retained; no automatic flashing
CUDA             source retained; GPU measurements not rerun

The published numerical claims remain tied to their documented hashes and
boundaries. A new or expanded claim requires the corresponding external run.
EOF
}

mode=${1:-local}
cd "$ROOT"

case "$mode" in
    quick)
        run_quick
        printf '%s\n' 'QUICK_VALIDATION=PASS'
        ;;
    local)
        run_local
        ;;
    status)
        show_status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

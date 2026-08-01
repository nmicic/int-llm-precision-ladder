#!/bin/sh
# Compare a rounded TinyLlama MGW directly with the exact Q16.48 oracle on
# prompts that are not part of the repository's four-prompt float gate.
set -eu

if [ "$#" -ne 7 ]; then
    echo "usage: $0 LLAMA_INT EXACT.mgw CANDIDATE.mgw REF_DIR PROMPTS.tsv RESULT_DIR LABEL" >&2
    exit 2
fi

BIN=$1
EXACT=$2
CANDIDATE=$3
REF_DIR=$4
PROMPTS=$5
RESULT_DIR=$6
LABEL=$7

mkdir -p "$RESULT_DIR"
SUMMARY="$RESULT_DIR/$LABEL-summary.txt"
: > "$SUMMARY"
MAX_NEW_TOKENS=20

sha256_file() {
    hash_path=$1
    if command -v sha256sum >/dev/null 2>&1; then
        hash_output=$(sha256sum "$hash_path") || return 1
    elif command -v shasum >/dev/null 2>&1; then
        hash_output=$(shasum -a 256 "$hash_path") || return 1
    else
        echo "neither sha256sum nor shasum is available" >&2
        return 1
    fi
    hash_value=${hash_output%%[[:space:]]*}
    case $hash_value in
        ''|*[!0-9a-fA-F]*)
            echo "invalid SHA-256 output for $hash_path" >&2
            return 1
            ;;
    esac
    if [ "${#hash_value}" -ne 64 ]; then
        echo "invalid SHA-256 length for $hash_path" >&2
        return 1
    fi
    printf '%s\n' "$hash_value"
}

extract_tokens() {
    input_log=$1
    output_tokens=$2
    awk '
        /^  Tokens: / {
            print
            count++
        }
        END {
            if (count != 1) {
                exit 1
            }
        }
    ' "$input_log" > "$output_tokens"
}

generated_count() {
    input_log=$1
    awk '
        /^[[:space:]]*Generated [0-9][0-9]* tokens in / {
            for (field = 1; field <= NF; field++) {
                if ($field == "Generated") {
                    value = $(field + 1)
                }
            }
            count++
        }
        END {
            if (count != 1) {
                exit 1
            }
            print value
        }
    ' "$input_log"
}

token_vector_count() {
    input_tokens=$1
    awk '
        {
            line = $0
            sub(/^  Tokens: \[/, "", line)
            sub(/\][[:space:]]*$/, "", line)
            count = split(line, tokens, ",")
            if (count < 1) {
                exit 1
            }
            for (field = 1; field <= count; field++) {
                gsub(/[[:space:]]/, "", tokens[field])
                if (tokens[field] !~ /^[0-9][0-9]*$/) {
                    exit 1
                }
            }
            print count
        }
    ' "$input_tokens"
}

binary_sha256=$(sha256_file "$BIN")
exact_sha256=$(sha256_file "$EXACT")
candidate_sha256=$(sha256_file "$CANDIDATE")
prompts_sha256=$(sha256_file "$PROMPTS")
{
    echo "# held-out integer-oracle gate"
    echo "# label $LABEL"
    echo "# binary_sha256 $binary_sha256"
    echo "# exact_sha256 $exact_sha256"
    echo "# candidate_sha256 $candidate_sha256"
    echo "# prompts_sha256 $prompts_sha256"
} >> "$SUMMARY"

passed=0
failed=0
matched_decisions=0
emitted_tokens=0
prompt_count=0
tab=$(printf '\t')
while IFS="$tab" read -r prompt_id prompt_text; do
    [ -n "$prompt_id" ] || continue
    prompt_count=$((prompt_count + 1))
    exact_log="$RESULT_DIR/$LABEL-$prompt_id-exact.txt"
    candidate_log="$RESULT_DIR/$LABEL-$prompt_id-candidate.txt"
    exact_tokens="$RESULT_DIR/$LABEL-$prompt_id-exact.tokens"
    candidate_tokens="$RESULT_DIR/$LABEL-$prompt_id-candidate.tokens"

    "$BIN" "$EXACT" --native --ref-dir "$REF_DIR" --generate \
        --prompt "$prompt_text" --max-new-tokens "$MAX_NEW_TOKENS" > "$exact_log"
    "$BIN" "$CANDIDATE" --native --ref-dir "$REF_DIR" --generate \
        --prompt "$prompt_text" --max-new-tokens "$MAX_NEW_TOKENS" > "$candidate_log"

    if ! extract_tokens "$exact_log" "$exact_tokens" || \
       ! extract_tokens "$candidate_log" "$candidate_tokens"; then
        echo "$prompt_id: FAIL (missing or duplicate token vector)" | tee -a "$SUMMARY"
        failed=$((failed + 1))
        continue
    fi
    if ! exact_generated=$(generated_count "$exact_log") || \
       ! candidate_generated=$(generated_count "$candidate_log"); then
        echo "$prompt_id: FAIL (missing or duplicate Generated count)" | tee -a "$SUMMARY"
        failed=$((failed + 1))
        continue
    fi
    if ! exact_token_count=$(token_vector_count "$exact_tokens") || \
       ! candidate_token_count=$(token_vector_count "$candidate_tokens"); then
        echo "$prompt_id: FAIL (malformed token vector)" | tee -a "$SUMMARY"
        failed=$((failed + 1))
        continue
    fi
    if [ "$exact_generated" -lt 1 ] || \
       [ "$candidate_generated" -lt 1 ] || \
       [ "$exact_generated" -gt "$MAX_NEW_TOKENS" ] || \
       [ "$candidate_generated" -gt "$MAX_NEW_TOKENS" ] || \
       [ "$exact_generated" -ne "$exact_token_count" ] || \
       [ "$candidate_generated" -ne "$candidate_token_count" ]; then
        echo "$prompt_id: FAIL (inconsistent Generated/token-vector count)" | tee -a "$SUMMARY"
        failed=$((failed + 1))
        continue
    fi

    if [ "$exact_generated" -eq "$candidate_generated" ] && \
       cmp -s "$exact_tokens" "$candidate_tokens"; then
        decisions=$exact_generated
        if [ "$exact_generated" -lt "$MAX_NEW_TOKENS" ]; then
            # The generator omits the terminal EOS from the displayed token
            # vector, but selecting it is still one evaluated greedy decision.
            decisions=$((decisions + 1))
        fi
        echo "$prompt_id: PASS (emitted=$exact_generated, evaluated_decisions=$decisions)" | tee -a "$SUMMARY"
        passed=$((passed + 1))
        emitted_tokens=$((emitted_tokens + exact_generated))
        matched_decisions=$((matched_decisions + decisions))
    else
        echo "$prompt_id: FAIL (generated exact=$exact_generated candidate=$candidate_generated)" | tee -a "$SUMMARY"
        diff -u "$exact_tokens" "$candidate_tokens" >> "$SUMMARY" || true
        failed=$((failed + 1))
    fi
done < "$PROMPTS"

maximum_requested=$((prompt_count * MAX_NEW_TOKENS))
echo "heldout: $passed passed, $failed failed, emitted=$emitted_tokens, matched_decisions=$matched_decisions, maximum_requested=$maximum_requested" | tee -a "$SUMMARY"
[ "$failed" -eq 0 ]

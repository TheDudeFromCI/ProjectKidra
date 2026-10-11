#!/bin/bash
# Usage: corpus.sh [--output PATH] [--pattern GLOB] INPUT [INPUT ...]

set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
source "./scripts/common.sh"

output=""
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --output)
            if [ $# -lt 2 ]; then
                echo "error: --output requires a path" >&2
                exit 1
            fi
            output="$2"
            shift 2
            ;;
        --output=*)
            output="${1#--output=}"
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

extra=()
[ -n "$output" ] && extra=(--output "$output")

python -m project_kidra.build_corpus "${args[@]}" "${extra[@]}"

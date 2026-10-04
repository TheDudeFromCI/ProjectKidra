#!/bin/bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
source "./scripts/common.sh"
set -e

python -m project_kidra "$@"

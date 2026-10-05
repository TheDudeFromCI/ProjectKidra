#!/bin/bash

set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
source "./scripts/common.sh"

mypy
pytest "$@"

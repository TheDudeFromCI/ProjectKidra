#!/bin/bash
# Installs dependencies, then type-checks and builds a distribution.

set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
source "./scripts/common.sh"

pip install -r requirements.txt
mypy
pip install build
python -m build

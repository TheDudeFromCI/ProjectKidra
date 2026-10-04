#!/bin/bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate

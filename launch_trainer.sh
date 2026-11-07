#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${STUDIO_BOOTSTRAP_PYTHON:-python3}" "$ROOT/tools/launch.py" --trainer "$@"

#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/FL-FD/fixed/run_upfall.sh" "$@"

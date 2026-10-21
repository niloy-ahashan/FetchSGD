#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/FL-FD/mhealth/run_mhealth_features.sh" "$@"

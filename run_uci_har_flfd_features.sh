#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/FL-FD/uci_har/run_uci_har_features.sh" "$@"

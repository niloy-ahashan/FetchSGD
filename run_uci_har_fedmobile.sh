#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/FedMobile/torch_port/run_uci_har_miss.sh" "$@"

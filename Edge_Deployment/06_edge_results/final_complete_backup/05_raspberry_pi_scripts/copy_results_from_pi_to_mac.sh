#!/usr/bin/env bash
set -euo pipefail
# Run on the MacBook to retrieve Raspberry Pi results.
PI_HOST="${1:-rockpi.local}"
PI_USER="${2:-pi}"
REMOTE_PROJECT="${3:-/home/$PI_USER/rock_edge_project}"
LOCAL_RESULTS="${4:-./raspberry_pi_results}"
mkdir -p "$LOCAL_RESULTS"
scp -r "$PI_USER@$PI_HOST:$REMOTE_PROJECT/06_edge_results" "$LOCAL_RESULTS/"
echo "Results copied to $LOCAL_RESULTS"

#!/usr/bin/env bash
set -euo pipefail
# Run on the MacBook after extracting 02_Rock_Burial_RaspberryPi_Edge_Deployment_FINAL_OPTIMIZED_RASPBERRY_PI_READY.zip.
PI_HOST="${1:-rockpi.local}"
PI_USER="${2:-pi}"
LOCAL_PROJECT="${3:-02_Rock_Burial_RaspberryPi_Edge_Deployment_FINAL_OPTIMIZED_RASPBERRY_PI_READY}"
REMOTE_PROJECT="${4:-/home/$PI_USER/rock_edge_project}"
scp -r "$LOCAL_PROJECT" "$PI_USER@$PI_HOST:$REMOTE_PROJECT"
echo "Copied to $PI_USER@$PI_HOST:$REMOTE_PROJECT"

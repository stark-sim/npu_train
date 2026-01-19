#!/bin/bash
# Robust background downloader with monitoring
# Supports resume, multiple retry strategies, and runs in background

# Load CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate npu_train

# Use proxy for external network access
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600

# Configuration
LOG_DIR="/home/sd/npu_train/download_logs"
PID_FILE="/home/sd/npu_train/.download.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# Start download in background
nohup python3 robust_download.py > "$LOG_DIR/$(date +%Y%m%d_%H%M%S)_stdout.log" 2>&1 &

# Save PID
PID=$!
echo $PID > "$PID_FILE"

echo "Background download started with PID: $PID"
echo "Log directory: $LOG_DIR"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_DIR/download_*.log"
echo ""
echo "Check status:"
echo "  cat /home/sd/npu_train/download_status.json"
echo ""
echo "Stop download:"
echo "  kill $PID"
echo "  # Or use: kill \$(cat $PID_FILE)"

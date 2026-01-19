#!/bin/bash
# Monitor and check download status

STATUS_FILE="/home/sd/npu_train/download_status.json"
PID_FILE="/home/sd/npu_train/.download.pid"
LOG_DIR="/home/sd/npu_train/download_logs"
MODELS_DIR="/home/sd/npu_train/models"
DATASETS_DIR="/home/sd/npu_train/datasets"

echo "=================================================================="
echo "          Download Status Report - $(date)"
echo "=================================================================="
echo ""

# Check if download is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "🟢 Download process is running (PID: $PID)"
    else
        echo "🔴 Download process is NOT running (stale PID file)"
    fi
else
    echo "⚪ No download process found"
fi
echo ""

# Show status from JSON file
if [ -f "$STATUS_FILE" ]; then
    echo "=================================================================="
    echo "                    Download Status from JSON"
    echo "=================================================================="
    cat "$STATUS_FILE"
    echo ""
else
    echo "No status file found yet..."
fi

echo ""
echo "=================================================================="
echo "                    Current Models on Disk"
echo "=================================================================="

# Check models
for model_dir in "$MODELS_DIR"/Qwen-Qwen2.5-*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
        safetensors_count=$(find "$model_dir" -name "*.safetensors" 2>/dev/null | wc -l)

        # Check if complete
        complete="⚠️ Incomplete"
        if [ "$safetensors_count" -gt 0 ] && [ -f "$model_dir/config.json" ] && [ -f "$model_dir/tokenizer.json" ]; then
            complete="✅ Complete"
        fi

        echo "$model_name"
        echo "  Size: $size"
        echo "  Safetensors files: $safetensors_count"
        echo "  Status: $complete"
        echo ""
    fi
done

echo ""
echo "=================================================================="
echo "                    Datasets Status"
echo "=================================================================="

if [ -d "$DATASETS_DIR" ]; then
    dataset_count=$(find "$DATASETS_DIR" -type d -mindepth 1 2>/dev/null | wc -l)
    dataset_size=$(du -sh "$DATASETS_DIR" 2>/dev/null | cut -f1)
    echo "Datasets directory exists"
    echo "  Datasets: $dataset_count"
    echo "  Total size: $dataset_size"
    echo ""
else
    echo "No datasets directory found"
fi

echo ""
echo "=================================================================="
echo "                    Disk Usage"
echo "=================================================================="
df -h /home/sd/npu_train 2>/dev/null | tail -1

echo ""
echo "=================================================================="
echo "                    Recent Logs"
echo "=================================================================="

if [ -d "$LOG_DIR" ]; then
    latest_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "Latest log: $latest_log"
        echo ""
        echo "Last 20 lines:"
        tail -20 "$latest_log"
        echo ""
    else
        echo "No logs found"
    fi
else
    echo "No log directory found"
fi

echo ""
echo "=================================================================="
echo "                    Commands"
echo "=================================================================="
echo "View full log:"
echo "  tail -f $LOG_DIR/download_*.log"
echo ""
echo "Check specific log:"
echo "  ls -lt $LOG_DIR/"
echo ""
echo "Stop download:"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "Restart download (resumable):"
echo "  ./start_download.sh"
echo ""

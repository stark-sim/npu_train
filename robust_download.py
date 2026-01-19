#!/usr/bin/env python3
"""
Robust background model downloader with resume capability and multiple retry strategies
Designed for poor network conditions
"""

import os
import sys
import time
import json
import signal
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add modelscope to path
sys.path.insert(0, '/home/sd/miniconda3/envs/npu_train/lib/python3.11/site-packages')

from modelscope.hub.snapshot_download import snapshot_download

# Configuration
BASE_DIR = "/home/sd/npu_train"
MODELS_DIR = f"{BASE_DIR}/models"
DATASETS_DIR = f"{BASE_DIR}/datasets"
LOG_DIR = f"{BASE_DIR}/download_logs"
STATUS_FILE = f"{BASE_DIR}/download_status.json"
LOCK_FILE = f"{BASE_DIR}/.download.lock"

# Models to download
MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "expected_size_gb": 3.0,
        "expected_files": 1,  # Single safetensors
        "local_name": "Qwen-Qwen2.5-1.5B-Instruct"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "expected_size_gb": 14.0,
        "expected_files": 4,  # 4 shards
        "local_name": "Qwen-Qwen2.5-7B-Instruct"
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "expected_size_gb": 28.0,
        "expected_files": 8,  # 8 shards
        "local_name": "Qwen-Qwen2.5-14B-Instruct"
    },
}

# Datasets to download
DATASETS = [
    "wikitext",
    "alpaca",
    "c4",  # Colossal Clean Crawler
]

# Download settings
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 30
TIMEOUT_SECONDS = 600  # 10 minutes per download attempt
MAX_CONCURRENT_DOWNLOADS = 2

# Setup logging
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DownloadStatus:
    """Track download status across runs"""

    def __init__(self):
        self.status = self._load_status()

    def _load_status(self):
        """Load status from file"""
        if Path(STATUS_FILE).exists():
            try:
                with open(STATUS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_status(self):
        """Save status to file"""
        with open(STATUS_FILE, 'w') as f:
            json.dump(self.status, f, indent=2)

    def is_model_completed(self, model_name):
        """Check if model download is complete"""
        local_name = MODELS[model_name]["local_name"]
        return self.status.get(local_name, {}).get("completed", False)

    def mark_model_completed(self, model_name, size_gb, files_count):
        """Mark model as completed"""
        local_name = MODELS[model_name]["local_name"]
        self.status[local_name] = {
            "completed": True,
            "completed_at": datetime.now().isoformat(),
            "size_gb": size_gb,
            "files_count": files_count
        }
        self.save_status()

    def is_dataset_completed(self, dataset_name):
        """Check if dataset download is complete"""
        return self.status.get(f"dataset_{dataset_name}", {}).get("completed", False)

    def mark_dataset_completed(self, dataset_name):
        """Mark dataset as completed"""
        self.status[f"dataset_{dataset_name}"] = {
            "completed": True,
            "completed_at": datetime.now().isoformat()
        }
        self.save_status()

    def get_summary(self):
        """Get summary of download status"""
        summary = {
            "models": {},
            "datasets": {}
        }
        for model_name, config in MODELS.items():
            local_name = config["local_name"]
            if local_name in self.status:
                summary["models"][local_name] = self.status[local_name]
        for dataset in DATASETS:
            key = f"dataset_{dataset}"
            if key in self.status:
                summary["datasets"][dataset] = self.status[key]
        return summary


def check_lock_file():
    """Check if another download is running"""
    lock_path = Path(LOCK_FILE)
    if lock_path.exists():
        # Check if it's stale (older than 2 hours)
        if time.time() - lock_path.stat().st_mtime > 7200:
            logger.warning("Found stale lock file, removing...")
            lock_path.unlink()
            return False
        logger.warning("Another download is running (lock file exists)")
        return True
    return False


def create_lock_file():
    """Create lock file"""
    Path(LOCK_FILE).touch()
    # Set up signal handler to remove lock on exit
    signal.signal(signal.SIGTERM, lambda s, f: Path(LOCK_FILE).unlink() if Path(LOCK_FILE).exists() else None)
    signal.signal(signal.SIGINT, lambda s, f: Path(LOCK_FILE).unlink() if Path(LOCK_FILE).exists() else None)


def remove_lock_file():
    """Remove lock file"""
    if Path(LOCK_FILE).exists():
        Path(LOCK_FILE).unlink()


def check_model_complete(model_name, config):
    """Check if model is fully downloaded"""
    local_name = config["local_name"]
    model_dir = Path(MODELS_DIR) / local_name

    if not model_dir.exists():
        return False, "Directory not found"

    # Check safetensors files
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if len(safetensors_files) != config["expected_files"]:
        return False, f"Expected {config['expected_files']} safetensors files, found {len(safetensors_files)}"

    # Check total size
    total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)
    if size_gb < config["expected_size_gb"] * 0.95:  # Allow 5% tolerance
        return False, f"Size {size_gb:.2f}GB is less than expected {config['expected_size_gb']:.2f}GB"

    # Check essential config files
    essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for ef in essential_files:
        if not (model_dir / ef).exists():
            return False, f"Missing essential file: {ef}"

    return True, "Complete"


def download_model_with_retry(model_name, config, status_tracker):
    """Download model with multiple retry strategies"""

    if status_tracker.is_model_completed(model_name):
        logger.info(f"Skipping {model_name} - already completed")
        return True

    # Check if already downloaded
    complete, reason = check_model_complete(model_name, config)
    if complete:
        logger.info(f"Model {config['local_name']} already complete: {reason}")
        status_tracker.mark_model_completed(model_name, config["expected_size_gb"], config["expected_files"])
        return True
    else:
        logger.info(f"Model {config['local_name']} incomplete: {reason}")

    local_name = config["local_name"]
    model_dir = Path(MODELS_DIR) / local_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Try different strategies
    strategies = ["modelscope", "wget", "aria2c"]

    for strategy in strategies:
        if strategy == "aria2c":
            # Check if aria2c is available
            try:
                subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
            except:
                logger.info(f"aria2c not available, skipping")
                continue
        elif strategy == "wget":
            try:
                subprocess.run(["wget", "--version"], capture_output=True, check=True)
            except:
                logger.info(f"wget not available, skipping")
                continue

        for attempt in range(MAX_RETRIES):
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} for {model_name} using {strategy}")

            try:
                if strategy == "modelscope":
                    success = download_model_modelscope(model_name, model_dir)
                elif strategy == "wget":
                    success = download_model_wget(model_name, model_dir, config)
                elif strategy == "aria2c":
                    success = download_model_aria2c(model_name, model_dir, config)

                if success:
                    complete, reason = check_model_complete(model_name, config)
                    if complete:
                        size_gb = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3)
                        status_tracker.mark_model_completed(model_name, size_gb, config["expected_files"])
                        logger.info(f"✅ Successfully downloaded {model_name} using {strategy}")
                        return True
                    else:
                        logger.warning(f"Download claimed success but incomplete: {reason}")

            except Exception as e:
                logger.error(f"Error downloading {model_name} with {strategy}: {e}")

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY_SECONDS * (attempt + 1)  # Exponential backoff
                logger.info(f"Waiting {delay}s before retry...")
                time.sleep(delay)

    logger.error(f"❌ Failed to download {model_name} after all strategies")
    return False


def download_model_modelscope(model_name, model_dir):
    """Download using ModelScope"""
    try:
        snapshot_download(
            model_name,
            cache_dir=MODELS_DIR,
            local_dir=model_dir,
            ignore_patterns=["*.bin", "*.pt", "*.onnx"],  # Only get safetensors
        )
        return True
    except Exception as e:
        logger.error(f"ModelScope download failed: {e}")
        raise


def download_model_wget(model_name, model_dir, config):
    """Download using wget with resume"""
    # Get file list from model info
    files = get_model_file_list(model_name)

    for file_path in files:
        url = f"https://www.modelscope.cn/{model_name}/resolve/master/{file_path}"
        local_path = model_dir / file_path

        # Skip if already exists and has correct size
        if local_path.exists():
            logger.info(f"File already exists: {file_path}")
            continue

        cmd = [
            "wget",
            "-c",  # Continue/resume
            "-t", "3",
            "--timeout=300",
            "--progress=bar:force",
            "--no-check-certificate",
            "-O", str(local_path),
            url
        ]

        logger.info(f"Downloading {file_path} with wget...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS * 2)

        if result.returncode != 0:
            logger.error(f"wget failed for {file_path}: {result.stderr}")
            raise Exception(f"wget failed for {file_path}")

    return True


def download_model_aria2c(model_name, model_dir, config):
    """Download using aria2c with resume"""
    files = get_model_file_list(model_name)

    # Create URL list for aria2c
    url_list = []
    for file_path in files:
        url = f"https://www.modelscope.cn/{model_name}/resolve/master/{file_path}"
        url_list.append(f"{url}\n  out={file_path}")

    # Write URL list to file
    url_file = model_dir / "urls.txt"
    with open(url_file, 'w') as f:
        f.write('\n'.join(url_list))

    cmd = [
        "aria2c",
        "-c",  # Continue
        "-x", "16",  # 16 connections per download
        "-s", "16",  # 16 segments per file
        "-j", str(MAX_CONCURRENT_DOWNLOADS),  # Max concurrent downloads
        "-t", "3",  # Max retries
        "--timeout=600",
        "--connect-timeout=60",
        "--max-tries=3",
        "--retry-wait=30",
        "-i", str(url_file),
        "-d", str(model_dir)
    ]

    logger.info(f"Downloading with aria2c ({len(files)} files)...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS * 10)

    if result.returncode != 0:
        logger.error(f"aria2c failed: {result.stderr}")
        raise Exception("aria2c failed")

    return True


def get_model_file_list(model_name):
    """Get list of files to download for a model"""
    files = []

    if "1.5B" in model_name:
        files.append("model.safetensors")
    elif "7B" in model_name:
        for i in range(1, 5):
            files.append(f"model-0000{i}-of-00004.safetensors")
    elif "14B" in model_name:
        for i in range(1, 9):
            files.append(f"model-0000{i}-of-00008.safetensors")

    # Add config files
    files.extend([
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json"
    ])

    return files


def download_datasets(status_tracker):
    """Download training datasets"""
    logger.info("\n" + "="*50)
    logger.info("Starting dataset downloads")
    logger.info("="*50 + "\n")

    from datasets import load_dataset

    for dataset_name in DATASETS:
        if status_tracker.is_dataset_completed(dataset_name):
            logger.info(f"Skipping {dataset_name} - already completed")
            continue

        for attempt in range(MAX_RETRIES):
            logger.info(f"Downloading dataset: {dataset_name} (attempt {attempt + 1}/{MAX_RETRIES})")

            try:
                dataset_dir = Path(DATASETS_DIR) / dataset_name
                dataset_dir.mkdir(parents=True, exist_ok=True)

                # Download dataset
                if dataset_name == "wikitext":
                    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=DATASETS_DIR)
                elif dataset_name == "alpaca":
                    dataset = load_dataset("tatsu-lab/alpaca", cache_dir=DATASETS_DIR)
                elif dataset_name == "c4":
                    dataset = load_dataset("allenai/c4", "en", cache_dir=DATASETS_DIR)

                status_tracker.mark_dataset_completed(dataset_name)
                logger.info(f"✅ Successfully downloaded dataset: {dataset_name}")
                break

            except Exception as e:
                logger.error(f"Error downloading dataset {dataset_name}: {e}")
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_SECONDS * (attempt + 1)
                    logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
        else:
            logger.error(f"❌ Failed to download dataset: {dataset_name}")


def main():
    """Main download function"""

    # Check lock file
    if check_lock_file():
        logger.error("Another download process is running. Exiting.")
        sys.exit(1)

    # Create lock file
    create_lock_file()

    try:
        # Initialize status tracker
        status = DownloadStatus()

        # Print current status
        logger.info("\n" + "="*50)
        logger.info("Current Download Status")
        logger.info("="*50)
        summary = status.get_summary()
        logger.info(json.dumps(summary, indent=2))

        # Download models
        logger.info("\n" + "="*50)
        logger.info("Starting Model Downloads")
        logger.info("="*50 + "\n")

        for model_name, config in MODELS.items():
            logger.info(f"\nProcessing: {model_name}")
            logger.info(f"Local name: {config['local_name']}")
            logger.info(f"Expected size: {config['expected_size_gb']}GB, Files: {config['expected_files']}")

            try:
                success = download_model_with_retry(model_name, config, status)
                if success:
                    logger.info(f"✅ {model_name} completed")
                else:
                    logger.error(f"❌ {model_name} failed")
            except Exception as e:
                logger.error(f"❌ Exception processing {model_name}: {e}")
                import traceback
                traceback.print_exc()

        # Download datasets
        try:
            download_datasets(status)
        except Exception as e:
            logger.error(f"Error downloading datasets: {e}")
            import traceback
            traceback.print_exc()

        # Final summary
        logger.info("\n" + "="*50)
        logger.info("Download Summary")
        logger.info("="*50)
        summary = status.get_summary()
        logger.info(json.dumps(summary, indent=2))

        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage(BASE_DIR)
        logger.info(f"\nDisk space: Used {used/(1024**3):.1f}GB, Free {free/(1024**3):.1f}GB")

    finally:
        # Remove lock file
        remove_lock_file()
        logger.info("\nDownload process completed")


if __name__ == "__main__":
    main()

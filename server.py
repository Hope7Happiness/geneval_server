from huggingface_hub import HfApi, hf_hub_download, upload_file
import time
import json
import os
import zipfile
import subprocess
from datetime import datetime
import traceback
import logging
from logging import StreamHandler, FileHandler
from huggingface_hub.utils import RepositoryNotFoundError


REPO_ID = "he-vision-group/geneval_server_test"
SERVER_ID = "h100-1"

api = HfApi()

# =============== Logging setup ===============
os.makedirs("server_logs", exist_ok=True)

LOG_FILE = "server_logs/output.log"
# each time restart server, remove old log file
if os.path.exists(LOG_FILE):
    os.rename(LOG_FILE, LOG_FILE + ".bak")

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",     # gray
        logging.INFO: "\033[36m",      # cyan
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[41m",  # white on red
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

logger = logging.getLogger("server")
logger.setLevel(logging.DEBUG)

# Console (colored)
console_handler = StreamHandler()
console_handler.setFormatter(ColorFormatter("[%(asctime)s] %(levelname)s: %(message)s"))

# File (no color)
file_handler = FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# ==============================================


def now():
    return datetime.utcnow().isoformat()


def download_info():
    logger.info("Downloading info.json ...")
    path = hf_hub_download(REPO_ID, "info.json", repo_type="dataset", force_download=True)
    with open(path) as f:
        return json.load(f)


def upload_info(obj):
    logger.info("Uploading updated info.json ...")
    upload_file(
        path_or_fileobj=json.dumps(obj, indent=2).encode(),
        path_in_repo="info.json",
        repo_id=REPO_ID,
        repo_type="dataset"
    )


while True:
    try:
        try:
            info = download_info()
        except RepositoryNotFoundError:
            logger.warning("No submission found. Waiting...")
            time.sleep(300)
            continue

        if info["status"] != "untested":
            logger.debug("Nothing to do. Status is %s", info["status"])
            time.sleep(300)
            continue

        logger.info("Claiming new task...")

        info["status"] = "running"
        info["claimed_by"] = SERVER_ID
        info["started_at"] = now()
        upload_info(info)
        
        # evaluating: run_id
        logger.info("Evaluating run_id=%s ...", info["run_id"])

        try:
            logger.info("Downloading submission.zip ...")
            zip_path = hf_hub_download(REPO_ID, "submission.zip", repo_type="dataset", force_download=True)

            if os.path.exists("workspace"):
                logger.warning("Cleaning existing workspace ...")
                os.system("rm -rf workspace")

            os.makedirs("workspace", exist_ok=True)

            with zipfile.ZipFile(zip_path) as z:
                z.extractall("workspace")

            # assert len(os.listdir("workspace")) == 1, f"Invalid submission content: {os.listdir('workspace')}"
            # submission_dir = os.path.join("workspace", os.listdir("workspace")[0])
            submission_dir = "workspace"

            os.remove(zip_path)

            logger.info("Running evaluation script on %s...", submission_dir)
            ret = subprocess.run(
                ["bash", "main.sh", submission_dir],
                capture_output=True,
                text=True
            )

            # save stdout/stderr as well
            with open("server_logs/stdout.log", "w") as f:
                f.write(ret.stdout)

            with open("server_logs/stderr.log", "w") as f:
                f.write(ret.stderr)

            if ret.returncode != 0:
                raise RuntimeError(f"main.sh failed: {ret.stderr}")

            logger.info("Uploading result.json ...")
            upload_file(
                path_or_fileobj="./server_logs/summary.json",
                path_in_repo="result.json",
                repo_id=REPO_ID,
                repo_type="dataset"
            )

        except Exception as e:
            logger.error("Evaluation failed: %s", e)

            info["status"] = "failed"
            info["finished_at"] = now()
            info["failure_reason"] = traceback.format_exc()

        else:
            logger.info("Evaluation finished successfully.")
            info["status"] = "tested"
            info["finished_at"] = now()
            info["failure_reason"] = None

        upload_info(info)

    except Exception as e:
        logger.critical("Server-level error: %s", e)

    logger.info("Sleeping before next check...")
    time.sleep(300) # check once per 10 min

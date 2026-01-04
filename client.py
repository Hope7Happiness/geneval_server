import json
import zipfile
import shutil
from datetime import datetime
import os
import time
from huggingface_hub import upload_file, hf_hub_download, delete_repo
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ID = "he-vision-group/geneval_server_test"
RUN_ID = "test_run"

def now():
    return datetime.utcnow().isoformat()

def submit_and_check(image_folder):
    # 1) build zip
    # recursive add all files in image_folder
    shutil.make_archive("submission", 'zip', image_folder)

    # we wait until the repo get deleted, to avoid concurrent submission issues
    for _ in range(20):
        try:
            old_info = hf_hub_download(REPO_ID, "info.json", repo_type="dataset", force_download=True)
            old_info = json.load(open(old_info))
        except RepositoryNotFoundError:
            break
        print(f"⏳ waiting for previous submission {old_info} to be evaluated ...")
        time.sleep(30)
        
    print("✓ previous submission cleared")

    upload_file(
        path_or_fileobj="submission.zip", 
        path_in_repo="submission.zip", 
        repo_id=REPO_ID, 
        repo_type="dataset"
    )
    print("✓ submission uploaded")
    
    # clean up local zip
    os.remove("submission.zip")

    # 2) create info.json
    info = {
        "run_id": RUN_ID,
        "status": "untested",
        "submitted_at": now(),
        "claimed_by": None,
        "started_at": None,
        "finished_at": None,
        "failure_reason": None,
    }


    upload_file(
        path_or_fileobj=json.dumps(info, indent=2).encode(),
        path_in_repo="info.json",
        repo_id=REPO_ID,
        repo_type="dataset"
    )
    print("✓ info.json uploaded")

    # 3) wait for result
    print("⏳ waiting for result ...")
    # time.sleep(600) # wait up to 10 minutes
    
    result = None
    for _ in range(20):
        path = hf_hub_download(REPO_ID, "info.json", repo_type="dataset", force_download=True)
        info = json.load(open(path))

        if info["status"] == "tested":
            print("✓ result ready")
            result = hf_hub_download(REPO_ID, "result.json", repo_type="dataset", force_download=True)
            # print("result path:", result)
            result = json.load(open(result))
            print("✓ result downloaded")
            break

        elif info["status"] == "failed":
            print("✗ evaluation failed:", info["failure_reason"])
            break
        
        elif info["status"] == "running":
            print("… still running ...")
        else:
            print(f"[WARNING] got unexpected status: {info['status']}")

        time.sleep(30)
        
    if result is None:
        print("✗ [FATAL] evaluation failed.")

    delete_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print("✓ repo deleted")
    
    return result
    
    # {
    #   "total_images": 2212,
    #   "total_prompts": 553,
    #   "percent_correct_images": 0.0,
    #   "percent_correct_prompts": 0.0,
    #   "task_scores": {
    #     "two_object": 0.0,
    #     "single_object": 0.0,
    #     "color_attr": 0.0,
    #     "position": 0.0,
    #     "colors": 0.0,
    #     "counting": 0.0
    #   },
    #   "overall_score": 0.0
    # }    

if __name__ == "__main__":
    submit_and_check("./random_out")
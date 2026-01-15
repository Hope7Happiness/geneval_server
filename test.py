# upload a fake file
from huggingface_hub import hf_hub_download, upload_file, create_repo, delete_repo

# create_repo("he-vision-group/geneval_server_test", repo_type="dataset", exist_ok=True)
# upload_file(
#     path_or_fileobj="test.py",
#     path_in_repo="test.py",
#     repo_id="he-vision-group/geneval_server_test",
#     repo_type="dataset",
#     token=True,
# )

# delete repo
delete_repo("he-vision-group/geneval_server_test", repo_type="dataset", token=True)
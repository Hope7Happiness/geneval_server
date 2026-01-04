from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhoukangyang/random_geneval",
    repo_type="dataset",
    local_dir="random_geneval",
    resume_download=True,
)

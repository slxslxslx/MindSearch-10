from huggingface_hub import snapshot_download
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="internlm/internlm2_5-7b-chat",
    local_dir="/nfs2/zdy_download/internlm2_5-7b-chat",
)

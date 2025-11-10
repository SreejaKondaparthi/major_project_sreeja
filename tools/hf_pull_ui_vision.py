from huggingface_hub import snapshot_download
from pathlib import Path

OUT = Path("data_raw/ui_vision")
OUT.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="ServiceNow/ui-vision",
    repo_type="dataset",
    local_dir=str(OUT),
    local_dir_use_symlinks=False # avoid symlink warnings on Windows
)
print("Downloaded UI-Vision into", OUT)

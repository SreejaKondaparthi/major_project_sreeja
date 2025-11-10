# tools/hf_pull_rico.py
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm

# Where to save raw files for the converter
ROOT = Path(".")
DST_IMG  = ROOT/"data_raw/rico/screens"
DST_JSON = ROOT/"data_raw/rico/view_hierarchies"
DST_IMG.mkdir(parents=True, exist_ok=True)
DST_JSON.mkdir(parents=True, exist_ok=True)

# Hugging Face dataset + config you saw on the page
HF_REPO  = "shunk031/Rico"
HF_NAME  = "ui-screenshots-and-view-hierarchies"

# Some repos offer multiple splits; we try a few common ones
CANDIDATE_SPLITS = ["train", "validation", "test", "all", "full"]

saved = 0
for split in CANDIDATE_SPLITS:
    try:
        ds = load_dataset(HF_REPO, name=HF_NAME, split=split, streaming=False)
    except Exception:
        continue

    print(f"Loaded split: {split}, length={len(ds)}")
    for i, row in enumerate(tqdm(ds)):
        # The image field may be "image" or "screenshot"
        img = row.get("image", None) or row.get("screenshot", None)
        vh  = row.get("view_hierarchy", None) or row.get("viewHierarchy", None)

        # Skip if anything is missing
        if img is None or vh is None:
            continue

        base = f"rico_{split}_{i:06d}"
        # Save PNG
        if isinstance(img, Image.Image):
            img.save(DST_IMG/f"{base}.png")
        else:
            # Sometimes it's an array-like image; convert via PIL
            Image.fromarray(img).save(DST_IMG/f"{base}.png")

        # Save JSON (string or dict)
        if isinstance(vh, (dict, list)):
            (DST_JSON/f"{base}.json").write_text(json.dumps(vh), encoding="utf-8")
        else:
            # assume string
            (DST_JSON/f"{base}.json").write_text(str(vh), encoding="utf-8")

        saved += 1

print(f"Saved {saved} image+json pairs into data_raw/rico/")

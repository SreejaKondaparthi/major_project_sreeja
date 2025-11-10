# tools/hf_stream_save_rico.py
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm

# grab a small batch first to test; set to None later to pull everything
MAX_PER_SPLIT = 50

REPO = "shunk031/Rico"
CFG  = "ui-screenshots-and-view-hierarchies"

ROOT = Path(__file__).resolve().parents[1]  # project root
DST_IMG  = ROOT / "data_raw" / "rico" / "screens"
DST_JSON = ROOT / "data_raw" / "rico" / "view_hierarchies"
DST_IMG.mkdir(parents=True, exist_ok=True)
DST_JSON.mkdir(parents=True, exist_ok=True)

def save_split(split: str) -> int:
    try:
        ds = load_dataset(REPO, name=CFG, split=split, streaming=True)
    except Exception as e:
        print(f"[{split}] can't load split: {e}")
        return 0

    saved = 0
    image_key = None
    vh_key = None

    for i, row in enumerate(tqdm(ds, desc=f"Saving {split}", unit="img")):
        if image_key is None:
            keys = row.keys()
            image_key = next((k for k in ("image","screenshot","img") if k in keys), None)

            # 👇 include 'activity' as the hierarchy key
            vh_key    = next((k for k in ("view_hierarchy","viewHierarchy","hierarchy","ui_hierarchy","activity") if k in keys), None)

            print(f"[{split}] detected columns -> image: {image_key}, vh: {vh_key}")
            if image_key is None or vh_key is None:
                print(f"[{split}] missing required columns. found keys: {list(keys)}")
                return 0

        base = f"rico_{split}_{i:06d}"
        img  = row[image_key]
        vh   = row[vh_key]

        # save image
        if hasattr(img, "save"):          # PIL.Image
            img.save(DST_IMG / f"{base}.png")
        else:                              # numpy array
            Image.fromarray(img).save(DST_IMG / f"{base}.png")

        # save json
        if isinstance(vh, (dict, list)):
            (DST_JSON / f"{base}.json").write_text(json.dumps(vh), encoding="utf-8")
        else:
            (DST_JSON / f"{base}.json").write_text(str(vh), encoding="utf-8")

        saved += 1
        if MAX_PER_SPLIT is not None and saved >= MAX_PER_SPLIT:
            break

    print(f"[{split}] saved {saved} pairs.")
    return saved

total = 0
for split in ("train","validation","test"):
    total += save_split(split)
print(f"Total saved: {total} PNG+JSON pairs into {DST_IMG.parent}")

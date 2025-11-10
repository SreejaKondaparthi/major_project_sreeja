from pathlib import Path
import json, itertools

root = Path("data_raw/ui_vision")
ann_files = list(root.glob("annotations/**/*.json")) + list(root.glob("annotations/**/*.jsonl"))
img_files = list(root.glob("images/**/*.*"))

print(f"Found {len(img_files)} images, {len(ann_files)} annotation files")
for p in ann_files[:10]:
    print("ANN:", p)

# peek first JSON file
for p in ann_files:
    try:
        if p.suffix == ".jsonl":
            line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            obj = json.loads(line)
        else:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        print("\nSample from:", p)
        if isinstance(obj, dict):
            print("dict keys:", list(itertools.islice(obj.keys(), 20)))
        elif isinstance(obj, list) and obj:
            print("list[0] keys:", list(obj[0].keys()))
        else:
            print("type:", type(obj))
        break
    except Exception as e:
        continue

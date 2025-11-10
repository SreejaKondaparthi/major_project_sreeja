import json, random, collections
from pathlib import Path
from PIL import Image

# ---- paths ----
UIV = Path("data_raw/ui_vision")
ANN = UIV / "annotations" / "element_grounding" / "element_grounding_basic.json"  # change to functional/spatial later if you want
IMG_ROOT = UIV / "images"  # we will find images recursively under here

YOLO = Path("data_yolo")
for s in ("train","val","test"):
    (YOLO/f"images/{s}").mkdir(parents=True, exist_ok=True)
    (YOLO/f"labels/{s}").mkdir(parents=True, exist_ok=True)

# ---- your 7 classes ----
NAMES = ['button','field','heading','image','label','link','text']
NAME_TO_ID = {n:i for i,n in enumerate(NAMES)}
IMG_EXTS = {".png",".jpg",".jpeg",".webp",".bmp"}

def canonical_label(raw: str) -> str:
    if not raw: return ""
    s = raw.lower()
    if any(k in s for k in ["button","btn","fab","submit","ok","next","save","apply","post","confirm","cancel"]): return "button"
    if any(k in s for k in ["input","field","textbox","search","email","password","username","query","edit","box"]): return "field"
    if any(k in s for k in ["title","header","heading","h1","h2","toolbar"]): return "heading"
    if any(k in s for k in ["image","img","icon","logo","avatar","thumbnail","thumb","imageview","iv","picture","pic"]): return "image"
    if any(k in s for k in ["label","subtitle","caption","hint","tag","chip","badge"]): return "label"
    if any(k in s for k in ["link","href","read_more","learn_more"]): return "link"
    if any(k in s for k in ["text","content","message","desc","value","body","paragraph"]): return "text"
    return ""

def build_image_index(root: Path):
    idx={}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx[p.name.lower()] = p
    return idx

def to_xyxy(b, W, H):
    """Accepts [x1,y1,x2,y2] or [x,y,w,h], absolute or normalized (<=1.2). Returns float x1,y1,x2,y2 in pixels."""
    if not isinstance(b,(list,tuple)) or len(b)<4:
        return None
    x1,y1,x2,y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

    # normalized?
    norm = max(x1,y1,x2,y2) <= 1.2

    # guess if it's xyxy (x2>x1 and y2>y1) else xywh
    is_xyxy = (x2 > x1) and (y2 > y1)

    if norm:
        if is_xyxy:
            x1,y1,x2,y2 = x1*W, y1*H, x2*W, y2*H
        else:
            # xywh normalized
            x2 = (x1 + x2) * W
            y2 = (y1 + y2) * H
            x1 = x1 * W
            y1 = y1 * H
    else:
        if not is_xyxy:
            # xywh absolute
            x2 = x1 + x2
            y2 = y1 + y2

    # clamp
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1,y1,x2,y2

def xyxy_to_yolo(x1,y1,x2,y2,W,H,cid):
    cx=(x1+x2)/2/W; cy=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def main(max_images=None, seed=0):
    random.seed(seed)

    if not ANN.exists():
        print("Annotation file not found:", ANN)
        return

    # 1) index all images by basename
    img_index = build_image_index(IMG_ROOT)
    if not img_index:
        print("No images found under", IMG_ROOT.resolve())
        return

    # 2) read the basic JSON (list of rows)
    data = json.loads(ANN.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, list):
        print("Unexpected JSON type. Expected a list of rows.")
        return

    # 3) group rows by image basename
    grouped = collections.defaultdict(list)
    unknown_labels = collections.Counter()
    missing_imgs = 0

    for row in data:
        img_path = row.get("image_path") or row.get("file_name") or row.get("image")
        bbox = row.get("bbox")
        lbl = row.get("element_type") or row.get("category")

        if not img_path or bbox is None: 
            continue

        base = Path(img_path).name.lower()
        if base not in img_index:
            missing_imgs += 1
            continue

        cname = canonical_label(str(lbl))
        if not cname:
            unknown_labels[str(lbl).lower()] += 1
            continue

        grouped[base].append({"label": cname, "bbox": bbox})

    if not grouped:
        print("No usable annotations. (All labels unmapped or images missing?)")
        print("Top unknown labels:", unknown_labels.most_common(20))
        return

    # 4) split by image 80/10/10
    keys = list(grouped.keys())
    if max_images:
        keys = keys[:max_images]
    random.shuffle(keys)
    n=len(keys); i_tr=int(0.8*n); i_va=int(0.9*n)
    split_keys = {
        "train": keys[:i_tr],
        "val": keys[i_tr:i_va],
        "test": keys[i_va:]
    }

    used=0
    for split, names in split_keys.items():
        for name in names:
            img_p = img_index[name]
            with Image.open(img_p) as im:
                W,H = im.size
            ylines=[]
            for o in grouped[name]:
                cid = NAME_TO_ID[o["label"]]
                xyxy = to_xyxy(o["bbox"], W, H)
                if not xyxy: 
                    continue
                line = xyxy_to_yolo(*xyxy, W,H,cid)
                if line: ylines.append(line)
            if not ylines:
                continue
            (YOLO/f"images/{split}/{img_p.name}").write_bytes(img_p.read_bytes())
            (YOLO/f"labels/{split}/{img_p.stem}.txt").write_text("\n".join(ylines), encoding="utf-8")
            used += 1

    counts = {s: len(list((YOLO/f"images/{s}").glob("*.*"))) for s in ("train","val","test")}
    print(f"Converted {used} UI-Vision images into data_yolo/")
    print("New split counts:", counts)
    if unknown_labels:
        print("Unmapped labels (top 20):", unknown_labels.most_common(20))
    if missing_imgs:
        print("Rows with missing images (by basename) skipped:", missing_imgs)

if __name__ == "__main__":
    # Start small: remove max_images to process all
    main(max_images=None)

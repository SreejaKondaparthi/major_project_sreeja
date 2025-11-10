import json, random
from pathlib import Path
from PIL import Image

# -------- paths --------
UIV = Path("data_raw/ui_vision")
ANN = UIV / "annotations" / "element_grounding" / "element_grounding_basic.json"  # change to functional/spatial if you want
IMG_ROOT = UIV / "images"  # we'll scan recursively

YOLO = Path("data_yolo")
for s in ("train","val","test"):
    (YOLO/f"images/{s}").mkdir(parents=True, exist_ok=True)
    (YOLO/f"labels/{s}").mkdir(parents=True, exist_ok=True)

# -------- class map (7 classes) --------
NAMES = ['button','field','heading','image','label','link','text']
NAME_TO_ID = {n:i for i,n in enumerate(NAMES)}

def canonical_label(raw: str) -> str:
    if not raw: return ""
    s = raw.lower()
    if any(k in s for k in ["button","btn","fab","submit","ok","next","save","apply","post","confirm","cancel"]): return "button"
    if any(k in s for k in ["input","field","textbox","search","email","password","username","query","edit","box"]): return "field"
    if any(k in s for k in ["title","header","heading","h1","h2","toolbar_title"]): return "heading"
    if any(k in s for k in ["image","img","icon","logo","avatar","thumbnail","thumb","imageview","iv","picture","pic"]): return "image"
    if any(k in s for k in ["label","subtitle","caption","hint","tag","chip","badge"]): return "label"
    if any(k in s for k in ["link","href","read_more","learn_more"]): return "link"
    if any(k in s for k in ["text","content","message","desc","value","body","paragraph"]): return "text"
    return ""

def xyxy_to_yolo(x1,y1,x2,y2,W,H,cid):
    # handle normalized coords if values are <= 1.2 (tolerate float noise)
    if max(x1,y1,x2,y2) <= 1.2:
        x1 *= W; x2 *= W; y1 *= H; y2 *= H
    # clamp
    x1=max(0,min(W-1,float(x1))); y1=max(0,min(H-1,float(y1)))
    x2=max(0,min(W-1,float(x2))); y2=max(0,min(H-1,float(y2)))
    if x2<=x1 or y2<=y1: return None
    cx=(x1+x2)/2/W; cy=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def parse_bbox(obj):
    # return (label, [x1,y1,x2,y2]) or None
    lbl = obj.get("label") or obj.get("class") or obj.get("category") or obj.get("name") or ""
    if "bbox" in obj and isinstance(obj["bbox"], (list,tuple)) and len(obj["bbox"]) >= 4:
        b = obj["bbox"]
        # heuristic: if b[2]>b[0] and b[3]>b[1] treat as xyxy; else assume xywh
        if b[2] > b[0] and b[3] > b[1]:
            x1,y1,x2,y2 = b[0], b[1], b[2], b[3]
        else:
            x1,y1,x2,y2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
        return lbl, [x1,y1,x2,y2]
    # x,y,w,h
    if all(k in obj for k in ("x","y")) and any(k in obj for k in ("w","width")) and any(k in obj for k in ("h","height")):
        w = obj.get("w", obj.get("width")); h = obj.get("h", obj.get("height"))
        return lbl, [obj["x"], obj["y"], obj["x"]+w, obj["y"]+h]
    # x1,y1,x2,y2
    if all(k in obj for k in ("x1","y1","x2","y2")):
        return lbl, [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]
    return None

def load_records(json_path: Path):
    js = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    # normalize to list of records with {image: <fname>, objects: [...]}
    if isinstance(js, dict):
        if "records" in js:
            data = js["records"]
        else:
            data = []
            for k,v in js.items():
                if isinstance(v, list):
                    data.append({"image": k, "objects": v})
    elif isinstance(js, list):
        data = js
    else:
        data = []

    recs = []
    for r in data:
        img = r.get("image") or r.get("image_name") or r.get("image_path") or r.get("file_name") or r.get("img") or r.get("img_name")
        if not img:
            # some schemas might nest like {"meta":{"image":...}}
            meta = r.get("meta") or {}
            img = meta.get("image") or meta.get("file_name")
        if not img:
            continue
        # normalize to basename
        img_name = Path(img).name
        objs = r.get("objects") or r.get("elements") or r.get("bboxes") or r.get("annotations") or r.get("boxes") or []
        std = []
        for o in objs:
            parsed = parse_bbox(o)
            if not parsed: continue
            std.append({"label": parsed[0], "bbox": parsed[1]})
        if std:
            recs.append({"image": img_name, "objects": std})
    return recs

def build_image_index(root: Path):
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in [".png",".jpg",".jpeg",".webp",".bmp"]:
            idx[p.name] = p
    return idx

def main(max_images=None, seed=0):
    random.seed(seed)

    # 1) image index
    img_index = build_image_index(IMG_ROOT)
    if not img_index:
        print("No images found under", IMG_ROOT)
        return

    # 2) load annotations
    recs = load_records(ANN)
    # keep only those we can resolve to a file
    recs = [r for r in recs if r["image"] in img_index]
    if max_images:
        recs = recs[:max_images]

    # 3) split 80/10/10
    random.shuffle(recs)
    n=len(recs); i_tr=int(0.8*n); i_va=int(0.9*n)
    splits = [("train", recs[:i_tr]), ("val", recs[i_tr:i_va]), ("test", recs[i_va:])]

    used = 0
    for split, items in splits:
        for r in items:
            img_path = img_index[r["image"]]
            with Image.open(img_path) as im:
                W,H = im.size
            lines=[]
            for o in r["objects"]:
                cname = canonical_label(o["label"])
                if not cname: 
                    continue
                cid = NAME_TO_ID[cname]
                x1,y1,x2,y2 = o["bbox"]
                line = xyxy_to_yolo(x1,y1,x2,y2,W,H,cid)
                if line: lines.append(line)
            if not lines:
                continue
            # write
            (YOLO/f"images/{split}/{img_path.name}").write_bytes(img_path.read_bytes())
            (YOLO/f"labels/{split}/{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            used += 1

    counts = {s: len(list((YOLO/f"images/{s}").glob("*.*"))) for s in ("train","val","test")}
    print(f"Converted {used} UI-Vision images into data_yolo/")
    print("Splits ->", counts)

if __name__ == "__main__":
    # start small to test; set to None to use all available
    main(max_images=500)

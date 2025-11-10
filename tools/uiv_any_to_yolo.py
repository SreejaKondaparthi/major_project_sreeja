import json, random
from pathlib import Path
from PIL import Image

UIV = Path("data_raw/ui_vision")
ANN_FILES = list(UIV.glob("annotations/**/*.json")) + list(UIV.glob("annotations/**/*.jsonl"))
IMG_ROOT = UIV / "images"

YOLO = Path("data_yolo")
for s in ("train","val","test"):
    (YOLO/f"images/{s}").mkdir(parents=True, exist_ok=True)
    (YOLO/f"labels/{s}").mkdir(parents=True, exist_ok=True)

NAMES = ['button','field','heading','image','label','link','text']
NAME_TO_ID = {n:i for i,n in enumerate(NAMES)}
IMG_EXTS = {".png",".jpg",".jpeg",".webp",".bmp"}

def canonical_label(raw):
    if not raw: return ""
    s = str(raw).lower()
    if any(k in s for k in ["button","btn","fab","submit","ok","next","save","apply","post","confirm","cancel"]): return "button"
    if any(k in s for k in ["input","field","textbox","search","email","password","username","query","edit","box"]): return "field"
    if any(k in s for k in ["title","header","heading","h1","h2","toolbar"]): return "heading"
    if any(k in s for k in ["image","img","icon","logo","avatar","thumbnail","thumb","imageview","iv","picture","pic"]): return "image"
    if any(k in s for k in ["label","subtitle","caption","hint","tag","chip","badge"]): return "label"
    if any(k in s for k in ["link","href","read_more","learn_more"]): return "link"
    if any(k in s for k in ["text","content","message","desc","value","body","paragraph"]): return "text"
    return ""

def xyxy_to_yolo(x1,y1,x2,y2,W,H,cid):
    # treat coords <=1.2 as normalized
    if max(x1,y1,x2,y2) <= 1.2:
        x1*=W; x2*=W; y1*=H; y2*=H
    x1=max(0,min(W-1,float(x1))); y1=max(0,min(H-1,float(y1)))
    x2=max(0,min(W-1,float(x2))); y2=max(0,min(H-1,float(y2)))
    if x2<=x1 or y2<=y1: return None
    cx=(x1+x2)/2/W; cy=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def parse_bbox(obj):
    # returns (label, [x1,y1,x2,y2]) or None
    lbl = obj.get("label") or obj.get("class") or obj.get("category") or obj.get("name") or obj.get("type") or ""
    # common shapes
    if "bbox" in obj and isinstance(obj["bbox"], (list,tuple)) and len(obj["bbox"])>=4:
        b=obj["bbox"]
        if b[2]>b[0] and b[3]>b[1]:  # xyxy
            return lbl, [b[0],b[1],b[2],b[3]]
        else:  # xywh
            return lbl, [b[0],b[1],b[0]+b[2],b[1]+b[3]]
    if all(k in obj for k in ("x","y")) and any(k in obj for k in ("w","width")) and any(k in obj for k in ("h","height")):
        w=obj.get("w",obj.get("width")); h=obj.get("h",obj.get("height"))
        return lbl, [obj["x"],obj["y"],obj["x"]+w,obj["y"]+h]
    if all(k in obj for k in ("x1","y1","x2","y2")):
        return lbl, [obj["x1"],obj["y1"],obj["x2"],obj["y2"]]
    if "coords" in obj and isinstance(obj["coords"], (list,tuple)) and len(obj["coords"])>=4:
        c=obj["coords"]; return lbl, [c[0],c[1],c[2],c[3]]
    return None

def to_records(any_json, file_hint=""):
    """
    Normalize MANY schema variants to [{image: <basename>, objects: [{label, bbox}, ...]}].
    """
    recs=[]
    if isinstance(any_json, dict):
        # possible keys containing list of records
        for k in ("records","data","items","annotations","images","entries"):
            if k in any_json and isinstance(any_json[k], list):
                for it in any_json[k]:
                    recs += to_records(it, file_hint)
                return recs
        # dict as one record?
        img = any_json.get("image") or any_json.get("image_name") or any_json.get("image_path") or any_json.get("file_name") or any_json.get("path") or any_json.get("screenshot")
        if not img and "meta" in any_json and isinstance(any_json["meta"], dict):
            m = any_json["meta"]
            img = m.get("image") or m.get("file_name") or m.get("path")
        objs = any_json.get("objects") or any_json.get("elements") or any_json.get("regions") or any_json.get("bboxes") or any_json.get("boxes") or any_json.get("labels") or any_json.get("annotations") or []
        if img and isinstance(objs, list):
            std=[]
            for o in objs:
                p = parse_bbox(o)
                if p: std.append({"label": p[0], "bbox": p[1]})
            if std:
                recs.append({"image": Path(str(img)).name, "objects": std})
        return recs
    if isinstance(any_json, list):
        for it in any_json:
            recs += to_records(it, file_hint)
        return recs
    return recs

def scan_images(root: Path):
    idx={}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx[p.name.lower()] = p
    return idx

def main(max_files=None, seed=0):
    random.seed(seed)
    if not ANN_FILES:
        print("No annotation files found under", (UIV/"annotations").resolve())
        return
    img_index = scan_images(IMG_ROOT)
    if not img_index:
        print("No images found under", IMG_ROOT)
        return

    all_recs=[]
    for f in ANN_FILES:
        try:
            if f.suffix==".jsonl":
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines()):
                    if not line.strip(): continue
                    obj = json.loads(line)
                    all_recs += to_records(obj, f.name)
                    if max_files and i>max_files: break
            else:
                obj = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
                all_recs += to_records(obj, f.name)
        except Exception as e:
            continue

    # keep only recs that we can map to an actual image
    recs = []
    seen = set()
    for r in all_recs:
        name = r["image"].lower()
        if name in img_index and (name, id(r)) not in seen and r["objects"]:
            recs.append({"image": img_index[name], "objects": r["objects"]})
            seen.add((name, id(r)))

    if not recs:
        print("Parsed 0 usable records. Likely the JSON schema uses different keys for image or boxes.")
        print("Run:  python tools/uiv_debug.py  and share the keys it prints.")
        return

    if max_files: recs = recs[:max_files]
    random.shuffle(recs)
    n=len(recs); i_tr=int(0.8*n); i_va=int(0.9*n)
    splits=[("train",recs[:i_tr]),("val",recs[i_tr:i_va]),("test",recs[i_va:])]

    used=0
    for split, items in splits:
        for r in items:
            img_p = r["image"]
            with Image.open(img_p) as im:
                W,H = im.size
            lines=[]
            for o in r["objects"]:
                cname = canonical_label(o.get("label",""))
                if not cname: 
                    continue
                cid = NAME_TO_ID[cname]
                x1,y1,x2,y2 = o["bbox"]
                line = xyxy_to_yolo(x1,y1,x2,y2,W,H,cid)
                if line: lines.append(line)
            if not lines: 
                continue
            (YOLO/f"images/{split}/{img_p.name}").write_bytes(img_p.read_bytes())
            (YOLO/f"labels/{split}/{img_p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            used+=1

    counts = {s: len(list((YOLO/f"images/{s}").glob("*.*"))) for s in ("train","val","test")}
    print(f"Converted {used} UI-Vision images into data_yolo/")
    print("Splits ->", counts)

if __name__ == "__main__":
    # start small; set to None for full run
    main(max_files=None)

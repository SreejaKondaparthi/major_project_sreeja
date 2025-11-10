# tools/rico_to_yolo.py
import json, re
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

PROJECT   = Path(__file__).resolve().parents[1]
RICO_IMG  = PROJECT/"data_raw/rico/screens"
RICO_JSON = PROJECT/"data_raw/rico/view_hierarchies"
OUT       = PROJECT/"data_yolo"

NAMES = ['button','field','heading','image','label','link','text']
NAME_TO_ID = {n:i for i,n in enumerate(NAMES)}

ANDROID_TO_CANON = {
    'button':'button','imagebutton':'button','appcompatbutton':'button','floatingactionbutton':'button',
    'edittext':'field','textinputedittext':'field','autocompletetextview':'field','appcompatedittext':'field',
    'textview':'text','appcompattextview':'text','imageview':'image',
}
CAND_CHILD_KEYS  = ("children","child","subviews","views","nodes","items","elements")
CAND_CLASS_KEYS  = ("klass","class","className","classname","type","widget","role")
CAND_BOUNDS_KEYS = ("bounds","rel_bounds","boundsInScreen","boundsInParent",
                    "absoluteBounds","visibleBounds","displayBounds","rect","frame","layout")
CAND_ID_KEYS     = ("resource_id","id","res_id")
CAND_DESC_KEYS   = ("content_desc","contentDescription","desc","description")

def norm_class_name(s: str) -> str:
    s = (s or "").lower()
    for key, canon in ANDROID_TO_CANON.items():
        if key in s: return canon
    return ""

def guess_from_text(s: str) -> str:
    if not s: return ""
    s = s.lower()
    if any(k in s for k in ["button","btn","fab"]): return "button"
    if any(k in s for k in ["edit","input","search","textbox","field","password"]): return "field"
    if any(k in s for k in ["title","header","toolbar","heading"]): return "heading"
    if any(k in s for k in ["image","img","icon","thumb","thumbnail","logo","avatar"]): return "image"
    if any(k in s for k in ["link","href"]): return "link"
    if any(k in s for k in ["label","subtitle","caption","hint"]): return "label"
    if any(k in s for k in ["text","message","description","content","value"]): return "text"
    return ""

def parse_bounds_any(b):
    if b is None: return None
    if isinstance(b, str):
        m = list(map(int, re.findall(r'-?\d+', b))); 
        return (m[0],m[1],m[2],m[3]) if len(m)>=4 else None
    if isinstance(b, (list,tuple)):
        if len(b)>=4 and all(isinstance(v,(int,float)) for v in b[:4]):
            x1,y1,x2,y2 = b[:4]; return (int(x1),int(y1),int(x2),int(y2))
        return None
    if isinstance(b, dict):
        kl = {k.lower() for k in b}
        if {"left","top","right","bottom"} <= kl:
            return (int(b["left"]),int(b["top"]),int(b["right"]),int(b["bottom"]))
        if {"x","y","width","height"} <= kl:
            x1,y1 = int(b["x"]),int(b["y"]); return (x1,y1,x1+int(b["width"]),y1+int(b["height"]))
        if {"x","y","w","h"} <= kl:
            x1,y1 = int(b["x"]),int(b["y"]); return (x1,y1,x1+int(b["w"]),y1+int(b["h"]))
        if "bounds" in b: return parse_bounds_any(b["bounds"])
        for k in ("boundsinscreen","boundsinparent","absolutebounds","visiblebounds","displaybounds"):
            for key in b.keys():
                if key.lower()==k: return parse_bounds_any(b[key])
    return None

def select_bounds_at_index(obj, idx):
    # try each candidate bounds list and pick a valid one at the same index
    for bk in CAND_BOUNDS_KEYS:
        for key in obj.keys():
            if key.lower()==bk:
                v = obj[key]
                if isinstance(v, (list,tuple)) and len(v)>idx:
                    bb = parse_bounds_any(v[idx])
                    if bb: return bb
    return None

def try_get_list(obj, keys):
    for k in keys:
        for key in obj.keys():
            if key.lower()==k:
                v = obj[key]
                return v if isinstance(v, list) else [v]
    return []

def to_yolo(x1,y1,x2,y2,w,h,cid):
    # clamp to image and convert
    x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
    if x2<=x1 or y2<=y1: return None
    cx = (x1+x2)/2/w; cy = (y1+y2)/2/h; bw = (x2-x1)/w; bh = (y2-y1)/h
    if bw<=0 or bh<=0 or (bw*bw+bh*bh) < 1e-6: return None
    return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def extract_json_boxes(obj, acc):
    if isinstance(obj, dict):
        # Columnar case check
        bounds_list = None
        klass_list  = None
        for bk in CAND_BOUNDS_KEYS:
            for key in obj.keys():
                if key.lower()==bk:
                    bounds_list = obj[key]; break
            if bounds_list is not None: break
        for ck in CAND_CLASS_KEYS:
            for key in obj.keys():
                if key.lower()==ck:
                    klass_list = obj[key]; break
            if klass_list is not None: break

        # If bounds is list-of-lists -> columnar
        if isinstance(bounds_list, list) and bounds_list and isinstance(bounds_list[0], (list,tuple)):
            n = len(bounds_list)
            rid_list  = try_get_list(obj, CAND_ID_KEYS)
            desc_list = try_get_list(obj, CAND_DESC_KEYS)
            for i in range(n):
                bb = select_bounds_at_index(obj, i)
                if not bb: continue
                x1,y1,x2,y2 = bb
                if x2<=x1 or y2<=y1: continue
                # class from klass[i] if possible
                raw_cls = ""
                if isinstance(klass_list, list) and i < len(klass_list):
                    raw_cls = klass_list[i]
                elif isinstance(klass_list, str):
                    raw_cls = klass_list
                cname = norm_class_name(raw_cls)
                # fallback heuristics from id/desc
                if not cname:
                    rid = (rid_list[i] if i < len(rid_list) else "") or ""
                    des = (desc_list[i] if i < len(desc_list) else "") or ""
                    cname = guess_from_text(str(rid)) or guess_from_text(str(des))
                if cname:
                    acc.append((cname, (x1,y1,x2,y2)))
        else:
            # Non-columnar dict node
            cls_str = None
            for ck in CAND_CLASS_KEYS:
                for key in obj.keys():
                    if key.lower()==ck:
                        v = obj[key]; cls_str = v[0] if isinstance(v, list) else v; break
                if cls_str is not None: break
            cname = norm_class_name(cls_str or "")
            # try bounds from any key
            bb = None
            for bk in CAND_BOUNDS_KEYS:
                for key in obj.keys():
                    if key.lower()==bk:
                        v = obj[key]
                        if isinstance(v, list) and v and isinstance(v[0], (list,tuple)):
                            v = v[0]
                        bb = parse_bounds_any(v)
                        if bb: break
                if bb: break
            if not cname:
                # heuristics from resource_id/content_desc
                rid_list  = try_get_list(obj, CAND_ID_KEYS)
                desc_list = try_get_list(obj, CAND_DESC_KEYS)
                guess = guess_from_text(str(rid_list[0]) if rid_list else "") or guess_from_text(str(desc_list[0]) if desc_list else "")
                if guess: cname = guess
            if cname and bb:
                x1,y1,x2,y2 = bb
                if x2>x1 and y2>y1:
                    acc.append((cname,(x1,y1,x2,y2)))

        # Recurse
        for v in obj.values():
            if isinstance(v, (dict, list)):
                extract_json_boxes(v, acc)

    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                extract_json_boxes(v, acc)

def extract_xml_boxes(xml_text, acc):
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return
    for el in root.iter():
        cls = el.attrib.get("class") or el.attrib.get("className") or ""
        cname = norm_class_name(cls)
        bb = parse_bounds_any(el.attrib.get("bounds"))
        if cname and bb:
            x1,y1,x2,y2 = bb
            if x2>x1 and y2>y1:
                acc.append((cname,(x1,y1,x2,y2)))

def infer_split_from_name(stem: str) -> str:
    s = stem.lower()
    if s.startswith("rico_train_"): return "train"
    if s.startswith("rico_validation_") or s.startswith("rico_valid_") or s.startswith("rico_val_"): return "val"
    if s.startswith("rico_test_"): return "test"
    return "train"

def main():
    for s in ('train','val','test'):
        (OUT/f"images/{s}").mkdir(parents=True, exist_ok=True)
        (OUT/f"labels/{s}").mkdir(parents=True, exist_ok=True)

    kept = 0
    for img_path in sorted(RICO_IMG.glob("*.*")):
        base = img_path.stem
        jpath = RICO_JSON/f"{base}.json"
        if not jpath.exists(): continue

        txt = jpath.read_text(encoding="utf-8", errors="ignore").strip()
        boxes = []

        # JSON first
        data = None
        try:
            data = json.loads(txt)
        except Exception:
            data = None

        if isinstance(data, dict):
            extract_json_boxes(data, boxes)
        elif isinstance(data, str):
            if "<" in data: extract_xml_boxes(data, boxes)
            else:
                try: extract_json_boxes(json.loads(data), boxes)
                except Exception: pass
        if not boxes and "<" in txt:
            extract_xml_boxes(txt, boxes)
        if not boxes: continue

        im = Image.open(img_path).convert("RGB")
        w,h = im.size
        lines=[]
        for cname,(x1,y1,x2,y2) in boxes:
            cid = NAME_TO_ID.get(cname); 
            if cid is None: continue
            yolo = to_yolo(x1,y1,x2,y2,w,h,cid)
            if yolo: lines.append(yolo)
        if not lines: continue

        split = infer_split_from_name(base)
        (OUT/f"images/{split}/{img_path.name}").write_bytes(img_path.read_bytes())
        (OUT/f"labels/{split}/{base}.txt").write_text("\n".join(lines), encoding="utf-8")
        kept += 1

    print(f"Converted {kept} RICO images into data_yolo/")

if __name__ == "__main__":
    main()

"""Build a clean, trainable baseline dataset from the raw 机车数据 dump.

- Merges obvious typo/case-duplicate class names.
- Keeps subtype classes separate (per decision).
- Keeps only "trainable" classes (>= thresholds); everything else (incl. crack)
  goes to a CANDIDATE set and is excluded from v1.
- Emits root/{images,labels,manifests} in the layout BaselineDetectionDataset expects.
- Images are hard-linked (no extra disk); labels are rewritten (cleaned).
- Produces a stratified train/val split.
"""
from __future__ import annotations
import os, glob, csv, shutil
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

SRC_ROOT = r"C:\Users\Zi Teng\Desktop\BRI\机车数据"
DST_ROOT = r"C:\Users\Zi Teng\Desktop\BRI\rt-detrv4\datasets\loco_v1"
VAL_FRAC = 0.10
MIN_IMAGES = 40       # a class needs at least this many images to be "trainable"
MIN_INSTANCES = 50    # ...and this many instances
SEED = 42
DROP_EMPTY_IMAGES = True   # drop images that have no kept boxes after filtering

# 1) typo / case duplicates -> canonical name
REMAP = {
    "LockSping": "LockSpring",
    "CotternPin": "CotterPin",
    "WarehouseSocket": "WareHouseSocket",
    "PortableLighterSocket": "PortableLightSocket",
}
# classes forced into the candidate set regardless of count (defect phase)
FORCE_CANDIDATE = {"crack"}


def canon(name: str) -> str:
    name = (name or "").strip()
    return REMAP.get(name, name)


def find_image(xml_path: str, filename: str):
    d = os.path.dirname(xml_path)
    for cand in (os.path.join(d, filename),
                 os.path.join(d, os.path.splitext(filename)[0] + ".jpg"),
                 os.path.join(d, os.path.splitext(os.path.basename(xml_path))[0] + ".jpg")):
        if cand and os.path.isfile(cand):
            return cand
    return None


def main():
    xmls = glob.glob(os.path.join(SRC_ROOT, "**", "*.xml"), recursive=True)
    print(f"found {len(xmls)} xml files")

    # ---- pass 1: count classes (after remap) at image level ----
    cls_imgs = defaultdict(set)
    cls_inst = Counter()
    for x in xmls:
        try:
            root = ET.parse(x).getroot()
        except Exception:
            continue
        for o in root.findall("object"):
            c = canon(o.findtext("name"))
            if not c:
                continue
            cls_imgs[c].add(x)
            cls_inst[c] += 1

    keep, candidate = [], []
    for c in cls_inst:
        if c in FORCE_CANDIDATE:
            candidate.append(c)
        elif len(cls_imgs[c]) >= MIN_IMAGES and cls_inst[c] >= MIN_INSTANCES:
            keep.append(c)
        else:
            candidate.append(c)
    keep.sort(key=lambda c: -cls_inst[c])
    candidate.sort(key=lambda c: -cls_inst[c])
    keep_set = set(keep)
    print(f"\nKEEP ({len(keep)}):")
    for c in keep:
        print(f"  {c:<24} inst={cls_inst[c]:<6} imgs={len(cls_imgs[c])}")
    print(f"\nCANDIDATE / excluded ({len(candidate)}): "
          + ", ".join(f"{c}({len(cls_imgs[c])}img)" for c in candidate))

    # ---- pass 2: build samples (only kept classes) ----
    samples = []  # (sample_id, image_path, [(cls,box)...])
    seen_ids = {}
    n_missing_img = 0
    for x in xmls:
        try:
            root = ET.parse(x).getroot()
        except Exception:
            continue
        objs = []
        for o in root.findall("object"):
            c = canon(o.findtext("name"))
            if c not in keep_set:
                continue
            bb = o.find("bndbox")
            if bb is None:
                continue
            box = (int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))),
                   int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax"))))
            objs.append((c, box))
        if DROP_EMPTY_IMAGES and not objs:
            continue
        fn = root.findtext("filename") or (os.path.splitext(os.path.basename(x))[0] + ".jpg")
        img = find_image(x, fn)
        if img is None:
            n_missing_img += 1
            continue
        sid = os.path.splitext(os.path.basename(x))[0]
        if sid in seen_ids:
            seen_ids[sid] += 1
            sid = f"{sid}__{seen_ids[sid]}"
        else:
            seen_ids[sid] = 0
        w = int(root.findtext("size/width") or 0)
        h = int(root.findtext("size/height") or 0)
        samples.append((sid, img, objs, w, h))

    print(f"\nusable images: {len(samples)} (missing image files skipped: {n_missing_img})")

    # ---- stratified split (greedy, rarest class first) ----
    import random
    rng = random.Random(SEED)
    rng.shuffle(samples)
    sid_classes = {s[0]: set(c for c, _ in s[2]) for s in samples}
    cls_to_sids = defaultdict(list)
    for sid, classes in sid_classes.items():
        for c in classes:
            cls_to_sids[c].append(sid)
    val_set = set()
    val_count = Counter()
    for c in sorted(keep, key=lambda c: len(cls_to_sids[c])):  # rarest first
        n = len(cls_to_sids[c])
        target = int(round(n * VAL_FRAC))
        if n >= 10:
            target = max(1, target)
        for sid in cls_to_sids[c]:
            if val_count[c] >= target:
                break
            if sid not in val_set:
                val_set.add(sid)
                for cc in sid_classes[sid]:
                    val_count[cc] += 1
    train_ids = [s[0] for s in samples if s[0] not in val_set]
    val_ids = [s[0] for s in samples if s[0] in val_set]
    print(f"split: train={len(train_ids)}  val={len(val_ids)}")

    # ---- write out ----
    img_dir = os.path.join(DST_ROOT, "images")
    lbl_dir = os.path.join(DST_ROOT, "labels")
    man_dir = os.path.join(DST_ROOT, "manifests")
    for d in (img_dir, lbl_dir, man_dir):
        os.makedirs(d, exist_ok=True)

    def link_or_copy(src, dst):
        if os.path.exists(dst):
            return
        try:
            os.link(src, dst)            # hardlink, no extra disk
        except Exception:
            shutil.copy2(src, dst)

    for sid, img, objs, w, h in samples:
        link_or_copy(img, os.path.join(img_dir, f"{sid}.jpg"))
        ann = ET.Element("annotation")
        ET.SubElement(ann, "filename").text = f"{sid}.jpg"
        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"
        for c, (x0, y0, x1, y1) in objs:
            ob = ET.SubElement(ann, "object")
            ET.SubElement(ob, "name").text = c
            ET.SubElement(ob, "difficult").text = "0"
            bb = ET.SubElement(ob, "bndbox")
            ET.SubElement(bb, "xmin").text = str(x0)
            ET.SubElement(bb, "ymin").text = str(y0)
            ET.SubElement(bb, "xmax").text = str(x1)
            ET.SubElement(bb, "ymax").text = str(y1)
        ET.ElementTree(ann).write(os.path.join(lbl_dir, f"{sid}.xml"), encoding="utf-8")

    with open(os.path.join(man_dir, "class_summary.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["class_name", "instances", "images"])
        for c in keep:
            wtr.writerow([c, cls_inst[c], len(cls_imgs[c])])
    # BaselineDetectionDataset resolves samples from a label path column
    # (one of relative_label_path/annotation_path/...), NOT from a sample_id.
    for name, ids in (("train_split.csv", train_ids), ("val_split.csv", val_ids)):
        with open(os.path.join(man_dir, name), "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(["relative_label_path"])
            for sid in ids:
                wtr.writerow([f"{sid}.xml"])

    print(f"\nDONE -> {DST_ROOT}")
    print(f"  classes={len(keep)}  images={len(samples)}  train={len(train_ids)}  val={len(val_ids)}")


if __name__ == "__main__":
    main()

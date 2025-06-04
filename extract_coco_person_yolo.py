import os
import json
import random
from tqdm import tqdm
from shutil import copy2
from collections import defaultdict

# === CONFIG ===
COCO_IMG_DIR = "./Coco/train2017"  # path to COCO images
COCO_JSON = "./Coco/annotations/instances_train2017.json"

OUTPUT_DIR = "coco_person_yolo"
TRAIN_RATIO = 0.9  # 90% train, 10% val

# Create output folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# Load COCO annotations
with open(COCO_JSON, 'r') as f:
    coco = json.load(f)

# Map image and size info
id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

# Get COCO category ID for "person"
person_cat_id = next(c["id"] for c in coco["categories"] if c["name"] == "person")

# Group person annotations by image
image_anns = defaultdict(list)
for ann in coco["annotations"]:
    if ann["category_id"] == person_cat_id and ann["iscrowd"] == 0:
        image_anns[ann["image_id"]].append(ann)

# Shuffle image IDs and split
image_ids = list(image_anns.keys())
random.shuffle(image_ids)
split_idx = int(len(image_ids) * TRAIN_RATIO)
train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])

# === Convert to YOLO format ===
for image_id, anns in tqdm(image_anns.items(), desc="Converting COCO to YOLO"):
    filename = id_to_filename[image_id]
    width, height = id_to_size[image_id]
    base = os.path.splitext(filename)[0]

    yolo_lines = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Determine split
    split = "train" if image_id in train_ids else "val"
    label_path = os.path.join(OUTPUT_DIR, "labels", split, base + ".txt")
    img_src = os.path.join(COCO_IMG_DIR, filename)
    img_dst = os.path.join(OUTPUT_DIR, "images", split, filename)

    # Write label
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # Copy image
    if os.path.exists(img_src):
        copy2(img_src, img_dst)

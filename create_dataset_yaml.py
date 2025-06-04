import os

# === CONFIG ===
dataset_root = './coco_person_yolo/'  # <-- CHANGE THIS
yaml_path = "person_cyclist.yaml"  # Output YAML filename

# Paths for images
train_dir = os.path.join(dataset_root, "images/train")
val_dir = os.path.join(dataset_root, "images/val")

# Class names
class_names = ["person", "cyclist"]

# YAML content
yaml_lines = [
    f"path: {dataset_root}",
    f"train: images/train",
    f"val: images/val",
    "",
    "names:"
]
yaml_lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]

# Write to file
with open(yaml_path, "w") as f:
    f.write("\n".join(yaml_lines))

print(f"✅ YAML file created: {yaml_path}")

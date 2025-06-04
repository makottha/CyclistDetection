import os

# === CONFIG ===
CYCLIST_LABEL_DIR = "cyclist_data_set/labels/train"  # or /val if needed
NEW_CLASS_ID = "1"
OLD_CLASS_ID = "0"

# Optional: update both train and val folders
for subfolder in ["train", "val"]:
    label_path = ['./cyclist_data_set/train/labels/', './cyclist_data_set/val/labels/']

for label in label_path:
    for file in os.listdir(label):
        if file.endswith(".txt"):
            file_path = label + file
            new_lines = []
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0] == OLD_CLASS_ID:
                        parts[0] = NEW_CLASS_ID
                    new_lines.append(" ".join(parts))

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))

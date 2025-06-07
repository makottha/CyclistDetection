import cv2
import os
from ultralytics import YOLO

# === CONFIGURATION ===
MODEL_FILE = "yolov8s.pt"
VIDEO_CLIPS_FOLDER_PATH = "Recorded_Videos"
OUTPUT_IMAGE_DIR = "train_data/images"
OUTPUT_LABEL_DIR = "train_data/labels"
FRAME_SKIP = 5
IOU_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.5
CLASS_NAMES = ["person", "cyclist"]
ANNOTATION_COLOR = (0, 255, 0)
ANNOTATION_THICKNESS = 2

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def calculate_iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area != 0 else 0

def get_cyclist_enclosing_box(person_box, bicycles, iou_threshold):
    best_iou = 0
    best_bicycle_box = None
    for bike_box in bicycles:
        iou = calculate_iou(person_box, bike_box)
        if iou > iou_threshold and iou > best_iou:
            best_iou = iou
            best_bicycle_box = bike_box
    if best_bicycle_box:
        x1 = min(person_box[0], best_bicycle_box[0])
        y1 = min(person_box[1], best_bicycle_box[1])
        x2 = max(person_box[2], best_bicycle_box[2])
        y2 = max(person_box[3], best_bicycle_box[3])
        return (x1, y1, x2, y2)
    return None

def save_yolo_annotation(frame, annotations, index, img_dir, lbl_dir):
    image_name = f"{index:05d}.jpg"
    label_name = f"{index:05d}.txt"
    image_path = os.path.join(img_dir, image_name)
    label_path = os.path.join(lbl_dir, label_name)
    if not cv2.imwrite(image_path, frame):
        print(f"[ERROR] Failed to save image: {image_path}")
        return False
    height, width = frame.shape[:2]
    try:
        with open(label_path, "w") as f:
            for cls_id, x1, y1, x2, y2 in annotations:
                xc = (x1 + x2) / 2 / width
                yc = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    except Exception as e:
        print(f"[ERROR] Failed to save label: {label_path}\n{e}")
        return False
    print(f"[SAVED] Frame {index} → {image_name} with {len(annotations)} objects")
    return True


def visualize_annotations(img_dir, lbl_dir):
    output_dir = os.path.join(os.path.dirname(img_dir), "human_readable_data")
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(img_dir):
        if not filename.endswith(".jpg"):
            continue
        image_path = os.path.join(img_dir, filename)
        label_path = os.path.join(lbl_dir, filename.replace(".jpg", ".txt"))
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        with open(label_path, "r") as f:
            for line in f:
                cls_id, xc, yc, w, h = map(float, line.strip().split())
                x1 = int((xc - w / 2) * width)
                y1 = int((yc - h / 2) * height)
                x2 = int((xc + w / 2) * width)
                y2 = int((yc + h / 2) * height)
                label = CLASS_NAMES[int(cls_id)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), ANNOTATION_COLOR, ANNOTATION_THICKNESS)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNOTATION_COLOR, 1)
        annotated_path = os.path.join(output_dir, filename)
        cv2.imwrite(annotated_path, frame)
    print(f"[INFO] Annotated images saved to: {output_dir}")

def process_video(video_path, model, image_index_start):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    total_saved = 0
    frame_count = 0
    image_index = image_index_start

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Skipping unreadable frame {frame_count} in {video_path}")
            break

        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        persons, bicycles = [], []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if conf < CONFIDENCE_THRESHOLD:
                continue
            label = model.names.get(class_id, "").lower()
            if "person" in label:
                persons.append((x1, y1, x2, y2))
            elif "bicycle" in label:
                bicycles.append((x1, y1, x2, y2))

        if not persons:
            frame_count += 1
            continue

        annotations = []
        for pbox in persons:
            cyclist_box = get_cyclist_enclosing_box(pbox, bicycles, IOU_THRESHOLD)
            if cyclist_box:
                annotations.append((1, *cyclist_box))
            else:
                annotations.append((0, *pbox))

        if annotations:
            if save_yolo_annotation(frame, annotations, image_index, OUTPUT_IMAGE_DIR, OUTPUT_LABEL_DIR):
                image_index += 1
                total_saved += 1

        frame_count += 1

    cap.release()
    print(f"[DONE] Processed {video_path}, saved {total_saved} annotated frames\n")
    return total_saved

if __name__ == "__main__":
    if not os.path.exists(VIDEO_CLIPS_FOLDER_PATH):
        print(f"[ERROR] Video folder not found: {VIDEO_CLIPS_FOLDER_PATH}")
        exit(1)

    video_files = [
        os.path.join(VIDEO_CLIPS_FOLDER_PATH, f)
        for f in os.listdir(VIDEO_CLIPS_FOLDER_PATH)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print(f"[WARN] No video files found in: {VIDEO_CLIPS_FOLDER_PATH}")
        exit(0)

    print(f"[INFO] Loading YOLO model: {MODEL_FILE}")
    model = YOLO(MODEL_FILE)

    total_images = 0
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[PROCESSING] ({idx}/{len(video_files)}) {video_path}")
        saved = process_video(video_path, model, total_images)
        total_images += saved

    print(f"\n✅ Finished processing all videos.")
    print(f"🖼️  Total annotated images saved: {total_images}")

    #visualize_annotations(OUTPUT_IMAGE_DIR, OUTPUT_LABEL_DIR)
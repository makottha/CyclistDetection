import cv2
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "./yolov8_person_cyclist/finetuned3/weights/best.pt"
VIDEO_PATH = "./video.mp4"
OUTPUT_PATH = "output_annotated_video.mp4"
CLASS_NAMES = ["person", "cyclist"]
CONF_THRESHOLD = 0.25

# === Color Map (BGR) ===
COLOR_MAP = {
    0: (0, 255, 0),     # green for person
    1: (0, 128, 255)    # orange for cyclist
}

# === Load model ===
model = YOLO(MODEL_PATH)

# === Load video ===
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Output video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    # Draw detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLOR_MAP.get(cls_id, (255, 255, 255))  # default white

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Annotated video saved to: {OUTPUT_PATH}")

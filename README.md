# 🚴‍♂️ YOLOv8 Fine-Tuned: Person and Cyclist Detection

This project fine-tunes a YOLOv8 object detection model to distinguish between **persons** and **cyclists** in images and videos. It combines COCO-derived data for `"person"` and a custom dataset for `"cyclist"`.

---

## 📂 Project Structure


---

## 🧠 Class Mapping

| Class ID | Class Name |
|----------|-------------|
| 0        | Person      |
| 1        | Cyclist     |

---

## 🚀 Getting Started

### 1. 📦 Install Requirements

```bash
pip install ultralytics opencv-python
```

```
python fine_tune_yolov8.py \
  --yaml person_cyclist.yaml \
  --model yolov8s.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --patience 10 \
  --name yolov8s_finetune
```

from ultralytics import YOLO
import argparse

def train_yolov8(yaml_path, model_name, epochs, imgsz, batch, project, run_name, lr0):
    # Load pretrained model
    model = YOLO(model_name)

    # Train
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=project,
        lr0=lr0,
        pretrained=True,
    )

    print(f"✅ Fine-tuning complete. Weights saved in 'runs/detect/{run_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on custom dataset")

    parser.add_argument("--yaml", type=str, default="person_cyclist.yaml", help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 pretrained model (e.g., yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", type=str, default="yolov8_person_cyclist", help="Project directory")
    parser.add_argument("--name", type=str, default="finetuned", help="Run name")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (in epochs)")

    args = parser.parse_args()

    train_yolov8(
        yaml_path=args.yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        run_name=args.name,
        lr0=args.lr0,
    )

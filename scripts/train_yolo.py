from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/workspace/Food-Ingredients-Detection-3/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    cache=False,
    project="/workspace/runs/detect/CampusChef",
    name="ingredient_detector"
)
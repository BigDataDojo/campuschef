from ultralytics import YOLO


model = YOLO("/workspace/yolov8x.pt")

model.train(
    data="/workspace/Recipe-Ingredients-2-3/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    workers=0,
    cache=False,
    device=0,
    project="/workspace/runs/detect/CampusChef",
    name="ingredient_detector_x"
)
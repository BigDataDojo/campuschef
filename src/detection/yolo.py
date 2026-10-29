from ultralytics import YOLO

MODEL_PATH = (
    "/workspace/runs/detect/CampusChef/"
    "cookai_ingredient_detector_x-2/weights/best.pt"
)

IMAGE_PATH = "/workspace/main/outputs/yolo/predictions/"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.8,
    save=True
)

for result in results:
    if len(result.boxes) == 0:
        print("No detections")
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.3f}")

from ultralytics import YOLO

model = YOLO("/workspace/runs/detect/CampusChef/ingredient_detector-4/weights/best.pt")

print(model.names)
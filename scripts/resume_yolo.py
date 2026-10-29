from ultralytics import YOLO

model = YOLO("/workspace/runs/detect/CampusChef/ingredient_detector-4/weights/last.pt")
model.train(resume=True)
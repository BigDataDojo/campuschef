import os

from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("food-w4zm1").project("recipe-ingredients-2")
version = project.version(3)
dataset = version.download("yolov8")
                
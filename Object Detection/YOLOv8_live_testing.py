from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

# Download the yolo model previously
model = YOLO("yolov8x.pt") # path to the model
results = model.predict(source="0", show=True, conf=0.7) # source = 0 specifies the source of image to predict (webcam)

print(results)
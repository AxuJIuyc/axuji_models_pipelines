from ultralytics import YOLO
import cv2

weights = "rknn/crossroads_roofs-yolov8m-1.2.3.4.9_12(best).pt" # "yolov8n.pt"
image = "../../../data/images/sat_1794919.jpg"
save = 'cuda/res.jpg'
device = 0 # 'cpu'

# Load a pretrained YOLO11n model
model = YOLO(weights)

# Perform object detection on an image
results = model(image, device=device, verbose=False)  # Predict on an image
results[0].show()  # Display results
cv2.imwrite(save, results[0].plot())
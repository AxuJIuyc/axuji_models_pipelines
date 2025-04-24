from ultralytics import YOLO
from time import monotonic
from tqdm import tqdm


def inference(data, device):
    return model(data, device=device, verbose=False)  # Predict on an image
    

# Load a pretrained YOLO11n model
weights = "rknn/crossroads_yolov8n.pt"
# device = 0
device = 'cpu'
image = "../../../data/images/sat_1794919.jpg"
N = 500

model = YOLO(weights)
t1 = monotonic()
for _ in tqdm(range(N)):
    inference(image, device)
t2 = monotonic()

dt = t2 - t1
d = dt/N
print(f"Performance ({device}):", round(d*10**3, 2), 'ms;', round(1/d, 2), 'fps')
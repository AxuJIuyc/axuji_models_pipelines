from rknn.api import RKNN
import numpy as np
import cv2

ONNX_MODEL_PATH = 'workdir/run_4/space_heat.onnx'
RKNN_MODEL_PATH = 'workdir/run_4/space_heat.rknn'

# Путь к изображению для теста (входной размер должен совпадать с моделью)
IMAGE_PATH = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1/train/images/002d_3.png"  # например 256x256 RGB
DATASET = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1/train/dataset.txt"

# Размер входа модели (должен соответствовать при экспорте из PyTorch)
INPUT_SIZE = (32, 32)  # H, W

# Создаем объект
rknn = RKNN()

# --> 1. Настройка сначала
print('--> Config')
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rv1103',  # для RV1103
    quantized_dtype='w8a8',
    optimization_level=3
)

# --> 2. Только теперь загрузка модели
print('--> Load ONNX model')
ret = rknn.load_onnx(model=ONNX_MODEL_PATH)
if ret != 0:
    print('Failed to load ONNX model!')
    exit(ret)

print('--> Step 3: Build RKNN model (with quantization)')
ret = rknn.build(do_quantization=True, dataset=DATASET)
if ret != 0:
    print('Build failed!')
    exit(ret)

print('--> Step 4: Export RKNN model')
ret = rknn.export_rknn(RKNN_MODEL_PATH)
if ret != 0:
    print('Failed to export RKNN model!')
    exit(ret)

print('--> Step 5: Load and test model on image')
ret = rknn.init_runtime()
if ret != 0:
    print('Runtime init failed')
    exit(ret)

# Препроцессинг изображения
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, INPUT_SIZE[::-1])  # W, H
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.uint8)

# Инференс
outputs = rknn.inference(inputs=[img])
output = outputs[0]

print('--> Inference done. Output shape:', output.shape)

rknn.release()

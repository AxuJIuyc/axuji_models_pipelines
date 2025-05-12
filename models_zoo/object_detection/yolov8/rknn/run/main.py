import os

import cv2
import numpy as np
from base import RK3588
from tqdm import tqdm

class RK3588_v2(RK3588):
    NMS_THRESH = 0.75
    OBJ_THRESH = 0.25

    def pre_process(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        img, ratio, dwdh = self.letterbox(img, auto=False, new_shape=(self.net_size, self.net_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        return img, ratio, dwdh

    def inference(self, img: np.ndarray) -> list[np.ndarray] | None:
        return self.rknn_lite.inference(inputs=[img])

    def filter_boxes(
        self,
        boxes: np.ndarray,
        box_confidences: np.ndarray,
        box_class_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.flatten()
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        scores = class_max_score * box_confidences
        mask = scores >= self.OBJ_THRESH

        return boxes[mask], classes[mask], scores[mask]

    def dfl(self, position: np.ndarray) -> np.ndarray:
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)

        exp_y = np.exp(y)
        y = exp_y / np.sum(exp_y, axis=2, keepdims=True)

        acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
        return np.sum(y * acc_metrix, axis=2)

    def box_process(self, position: np.ndarray) -> np.ndarray:
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w)
        stride = np.array([self.net_size // grid_h, self.net_size // grid_w]).reshape(
            1, 2, 1, 1
        )

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def post_process(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

        defualt_branch = 3
        pair_per_branch = len(outputs) // defualt_branch

        boxes, classes_conf, scores = [], [], []
        for i in range(defualt_branch):
            boxes.append(self.box_process(outputs[pair_per_branch * i]))
            classes_conf.append(sp_flatten(outputs[pair_per_branch * i + 1]))
            scores.append(np.ones_like(classes_conf[-1][:, :1], dtype=np.float32))

        boxes = np.concatenate([sp_flatten(b) for b in boxes])
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores).flatten()

        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.OBJ_THRESH, self.NMS_THRESH
        )
        if isinstance(indices, tuple):
            return None, None, None

        indices = [i for i in indices if scores[i] > self.OBJ_THRESH]
    
        boxes = boxes[indices]
        classes = classes[indices]
        scores = scores[indices]

        return boxes, classes, scores


if __name__ == "__main__":
    import time
    model_name = "/home/orangepi/PavelZh/roofs/crossroads_yolov8n_i8.rknn"
    # model_name = "/home/orangepi/PavelZh/roofs/crossroads_fp.rknn"
    model_name = "/home/orangepi/PavelZh/crossroads_roofs-yolov8m-1.12.2.3.4.9_0.rknn"
    # model_path = os.path.join(os.path.dirname(__file__), "models", model_name)
    model_path = model_name
    source = "/home/orangepi/PavelZh/roofs/Iskitim_1_500_256x256_223.jpeg"
    source = "/home/orangepi/PavelZh/images/images/sat_1757675.jpg"

    rk3588 = RK3588_v2(model_path, net_size=256)

    N = 1
    t1 = time.time()
    for i in tqdm(range(N)):
        rk3588.main(source,draw=True)
    t2 = time.time()
    print(f"FPS: {N / (t2 - t1):.2f}")
    print(f"Time: {1000*(t2-t1)/N:.1f} ms")

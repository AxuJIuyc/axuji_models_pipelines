import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm
from time import monotonic


class YOLOOpenvino:
    CONF_TH = 0.5
    """The confidence threshold for the results of the onnx model."""
    IOU_TH = 0.5
    """The threshold of intersection over union for the results of the onnx model."""

    def __init__(self, model_path: str, net_size=640):
        core = ov.Core()
        rec_model = core.read_model(model=model_path)
        self.yolo = core.compile_model(model=rec_model, device_name="CPU")
        """An openvino compilation to launch the model."""
        self.yolo_output_layer = self.yolo.output(0)
        """The name of the output layer."""
        self.net_size = net_size
        """Size of yolo input image"""

    @staticmethod
    def letterbox(
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleup=True,
        stride=32,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """
        Resizes and pads an image to fit a new shape while maintaining aspect ratio.

        Parameters
        ----------
        im : np.ndarray
            The input image to be resized and padded.
        new_shape : tuple[int, int], optional
            The desired output shape (height, width). Default is (640, 640).
        color : tuple[int, int, int], optional
            The color for padding. Default is (114, 114, 114).
        auto : bool, optional
            If True, adjusts padding to be a multiple of stride. Default is True.
        scaleup : bool, optional
            If True, allows scaling up the image. If False, only scales down. Default is True.
        stride : int, optional
            The stride for padding adjustment. Default is 32.

        Returns
        -------
        tuple[np.ndarray, float, tuple[float, float]]
            A tuple containing the resized and padded image,
            the scaling ratio, and the padding values.
        """

        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, r, (dw, dh)

    def pre_process(
        self, input_img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        img = input_img.copy()
        img, ratio, dwdh = self.letterbox(img, new_shape=[self.net_size, self.net_size], auto=False)
        img = np.expand_dims(img, axis=0).astype("float32") / 255.0
        img = np.transpose(img, [0, 3, 1, 2])

        return img, ratio, dwdh

    def inference(self, img: np.ndarray) -> np.ndarray:
        return self.yolo([img])[self.yolo_output_layer]

    def post_process(
        self, output: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        outputs = np.transpose(np.squeeze(output))
        classes_scores = outputs[:, 4:]
        max_scores = np.amax(classes_scores, axis=1)
        conf_indices = np.where(max_scores >= self.CONF_TH)[0]

        boxes, classes, scores = [], [], []

        for i in conf_indices:
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            x0 = x - w / 2
            x1 = x + w / 2
            y0 = y - h / 2
            y1 = y + h / 2

            boxes.append([x0, y0, x1, y1])
            classes.append(class_id)
            scores.append(max_score)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.CONF_TH, self.IOU_TH)
        
        if isinstance(indices, tuple):
            return None, None, None

        boxes = np.array(boxes)[indices]
        classes = np.array(classes)[indices]
        scores = np.array(scores)[indices]

        return boxes, classes, scores
    
    def predict(self, img: np.ndarray) -> np.ndarray:
        img, ratio, dwdh = self.pre_process(img)
        outputs = self.inference(img)
        if outputs is not None:
            boxes, classes, scores = self.post_process(outputs)
            if boxes is not None:
                boxes -= np.array(dwdh * 2)
                boxes /= ratio
                boxes = boxes.round().astype(np.int32)

                boxes, classes, scores = (
                    np.array(boxes), 
                    np.reshape(classes, (len(classes), -1)), 
                    np.reshape(scores, (len(scores), -1)))
                return np.concatenate((boxes, scores, classes), axis=1)
        return None
    
    @staticmethod
    def draw(img, res, save=False):
        im = img.copy()
        if res is not None:
            boxes = res[:,:4].astype(np.int32)
            for box in boxes:
                im = cv2.rectangle(
                    im, 
                    box[:2], box[2:4],
                    color=(0,0,250), 
                    thickness=2
                )
            if save:
                cv2.imwrite(save, im)
        else:
            print("No detections")
        return im
    
if __name__ == "__main__":
    modelpath = "crossroads_yolov8n.xml"
    net_size=256
    imgpath = "../../../../data/images/sat_1794919.jpg"
    save = "res.jpg"
    N = 500
    
    model = YOLOOpenvino(model_path=modelpath, net_size=net_size)
    
    x1,y1 = 400, 500
    img = cv2.imread(imgpath)[y1:y1+net_size,x1:x1+net_size]
    res = model.predict(img)
    img = model.draw(img, res, save=save)
    
    t1 = monotonic()
    for _ in tqdm(range(N)):
        img = cv2.imread(imgpath)[y1:y1+net_size,x1:x1+net_size]
        res = model.predict(img)
    t2 = monotonic()
    dt = t2-t1
    print(f'Time per frame: {dt*10**3/N} ms; FPS: {N/dt}')
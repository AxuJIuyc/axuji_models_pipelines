import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from time import monotonic

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
                               Each box is represented as [x1, y1, x2, y2].
        scores (numpy.ndarray): Array of scores with shape (N,).
        iou_threshold (float): Intersection-over-Union (IoU) threshold for suppression.

    Returns:
        numpy.ndarray: Indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # Compute areas of all boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by scores in descending order
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        # Select the box with the highest score and add it to the keep list
        i = order[0]
        keep.append(i)

        # Compute IoU of the selected box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection

        iou = intersection / union

        # Keep boxes with IoU less than the threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return np.array(keep, dtype=int)

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

def init(model_path, providers):
    """An onnx session to launch the model."""
    session = ort.InferenceSession(
            model_path, 
            providers=providers
        )
    """The name of the input metadata."""
    input_name: str = session.get_inputs()[0].name
    """The name of the output metadata."""
    output_name: list[str] = [session.get_outputs()[0].name]
    return session, input_name, output_name

def pre_process(
        input_img: np.ndarray, 
        new_shape=(256,256)
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Args:
        input_img (np.ndarray): image array
        new_shape (tuple, optional): (Height, Width). Defaults to (256,256).

    Returns:
        tuple[np.ndarray, float, tuple[float, float]]: _description_
    """
    img = input_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, ratio, dwdh = letterbox(img, new_shape=new_shape, auto=False)
    img = np.expand_dims(img, axis=0).astype("float32") / 255.0
    img = np.transpose(img, [0, 3, 1, 2])

    return img, ratio, dwdh

def inference(
    img: np.ndarray,
    session,
    input_name,
    output_name
    ) -> np.ndarray:
    return session.run(output_name, {input_name: img})[0]

def post_process(
        output: np.ndarray, 
        dwdh: tuple, 
        ratio: float,
        CONF_TH=0.3,
        IOU_TH=0.7
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        outputs = np.transpose(np.squeeze(output))
        classes_scores = outputs[:, 4:]
        max_scores = np.amax(classes_scores, axis=1)  # Max score for each prediction
        conf_indices = np.where(max_scores >= CONF_TH)[0]  # Filter based on CONF_TH

        if len(conf_indices) == 0:  # Early exit if no valid detections
            return None, None, None

        # Filter boxes and scores for confident detections
        filtered_outputs = outputs[conf_indices]
        filtered_scores = classes_scores[conf_indices]
        max_scores = max_scores[conf_indices]
        class_ids = np.argmax(filtered_scores, axis=1)  # Class IDs for confident detections
        boxes = filtered_outputs[:, :4]

        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x0 = x - w / 2
        x1 = x + w / 2
        y0 = y - h / 2
        y1 = y + h / 2

        boxes = np.stack([x0, y0, x1, y1], axis=1)  # Bounding boxes
        scores = max_scores  # Max scores for filtered detections

        # Compute bounding box coordinates
        boxes -= np.array(dwdh * 2)
        boxes /= ratio
        boxes = boxes.round().astype(np.int32)

        # Perform Non-Maximum Suppression
        indices = non_maximum_suppression(boxes, scores, IOU_TH)
        # indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.CONF_TH, self.IOU_TH)

        if isinstance(indices, tuple) or len(indices) == 0:  # No detections after NMS
            return None, None, None

        indices = np.array(indices).flatten()  # Flatten indices
        boxes = boxes[indices]
        classes = class_ids[indices]
        scores = scores[indices]

        return boxes, classes, scores
    
def draw(im, res, save=False):
    im = im.copy()
    for box in res:
        cv2.rectangle(im, box[:2], box[2:], color=(0,0,250), thickness=2)
    if save:
        cv2.imwrite(save, im)
    return im

if __name__ == "__main__":
    model_path = "crossroads_yolov8m.onnx"
    impath = "../../../../data/images/sat_1794919.jpg"
    N = 500
    providers = [
                "CUDAExecutionProvider", 
                "CPUExecutionProvider"
            ]
    
    session, input_name, output_name = init(model_path, providers)
    image = cv2.imread(impath)
    im, ratio, dwdh = pre_process(image)
    pred = inference(im, session, input_name, output_name)
    boxes, classes, scores = post_process(pred, dwdh, ratio, CONF_TH=0.3, IOU_TH=0.7)

    image = cv2.imread(impath)
    t1 = monotonic()
    for _ in tqdm(range(N)):
        im, ratio, dwdh = pre_process(image)
        pred = inference(im, session, input_name, output_name)
        boxes, classes, scores = post_process(pred, dwdh, ratio, CONF_TH=0.3, IOU_TH=0.7)
        # draw(image, boxes, save='res.jpg')
    t2 = monotonic()
    dt = t2-t1
    print(f'Time per frame: {dt*10**3/N} ms; FPS: {N/dt}')
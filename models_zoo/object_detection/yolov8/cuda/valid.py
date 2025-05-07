import os
from tabnanny import verbose
from ultralytics import YOLO
from tqdm import tqdm


def validation(folder, model):
    """_summary_

    Args:
        folder (str): folder with data folders (images + labels)
        model (str): path to model file.pt
    """
    # Init model
    yolo = YOLO(model)
    # get folders
    folders = get_folders(folder)
    # get data
    data = get_data(folders)
    # valid
    # preds, gts = [], []
    metrics = []
    for im, lbl in tqdm(data, desc='Validation'):
        res = yolo(im, verbose=False)[0]
        h,w = res.orig_shape # H, W
        pred = res.boxes.data.cpu().numpy()
        # pred = res.boxes.data.cpu().tolist()
        
        gt = np.array(read_anno(lbl))
        if len(gt) == 0:
            pass
        else:
            gt[:,1], gt[:,3] = gt[:,1]*w, gt[:,3]*w
            gt[:,2], gt[:,4] = gt[:,2]*h, gt[:,4]*h
        
        # preds.append(pred)
        # gts.append(gt)
        
        #calculate metrics
        metrics.append(compute_metrics(gt, pred))
        
    metrics = sum_metrics(metrics)
    # print(metrics)
    # p = sum([x['precision'] for x in metrics])/len(metrics)
    # r = sum([x['recall'] for x in metrics])/len(metrics)
    # m50 = sum([x['map50'] for x in metrics])/len(metrics)
    # m5095 = sum([x['map50_95'] for x in metrics])/len(metrics)
    # return p, r, m50, m5095
    return metrics  
    
def get_folders(folder):
    files = os.listdir(folder)
    folders = []
    for file in files:
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            folder_files = os.listdir(path)
            if 'images' in folder_files and 'labels' in folder_files:
                folders.append(path)
    return folders
    
def get_data(folders):
    data = []
    for path in folders:
        images_path = os.path.join(path, 'images', 'train') 
        labels_path = os.path.join(path, 'labels', 'train')
        images = sorted(os.listdir(images_path)) 
        labels = sorted(os.listdir(labels_path))
        for im in images:
            name, ext = os.path.splitext(im)
            if name+'.txt' in labels:
                data.append([
                    os.path.join(images_path, im),
                    os.path.join(labels_path, name+'.txt'),
                ])
    return data

def read_anno(txt):
    with open(txt, 'r') as f:
        text = f.read().splitlines()
    anno = []
    for row in text:
        if len(row) > 3:
            lbl, x,y,w,h = map(float, row.split())
            anno.append([lbl, x,y,w,h])
    return anno

# ========= Calculate metrics --> =========================
# /metrics/object_detection_metrics.py
from typing import List, Tuple, Dict
import numpy as np
# from collections import defaultdict, Counter


BBox = Tuple[float, float, float, float, float]  # (label, x, y, w, h)
PredBBox = Tuple[float, float, float, float, float, float]  # (x, y, w, h, score, label)

def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Computes the IoU matrix between ground-truth and predicted bounding boxes.
    :param gt_boxes: (N, 4) ndarray of gt boxes in format (x, y, w, h)
    :param pred_boxes: (M, 4) ndarray of predicted boxes in format (x, y, w, h)
    :return: (N, M) IoU matrix
    """
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]))
    gt_boxes = gt_boxes.reshape(-1, 4)
    pred_boxes = pred_boxes.reshape(-1, 4)

    gt_x1, gt_y1, gt_w, gt_h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

    gt_x2 = gt_x1 + gt_w
    gt_y2 = gt_y1 + gt_h
    pred_w, pred_h = pred_x2-pred_x1, pred_y2-pred_y1
    # pred_x2 = pred_x1 + pred_w
    # pred_y2 = pred_y1 + pred_h

    # Broadcasting boxes for vectorized computation
    inter_x1 = np.maximum(gt_x1[:, None], pred_x1[None, :])
    inter_y1 = np.maximum(gt_y1[:, None], pred_y1[None, :])
    inter_x2 = np.minimum(gt_x2[:, None], pred_x2[None, :])
    inter_y2 = np.minimum(gt_y2[:, None], pred_y2[None, :])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    gt_area = gt_w * gt_h
    pred_area = pred_w * pred_h

    union_area = gt_area[:, None] + pred_area[None, :] - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


def compute_confusion_matrix(
    gt_labels: List[str],
    pred_labels: List[str],
    pred_scores: List[float],
    iou_matrix: np.ndarray,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, Dict[str, float]]]:
    classes = sorted(set(gt_labels) | set(pred_labels))
    class_to_idx = {int(cls): i for i, cls in enumerate(classes)}
    num_classes = len(classes)

    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)  # +1 for background

    if len(gt_labels) == 0 and len(pred_labels) == 0:
        stats = {k: 0.0 for k in ['tp', 'fp', 'fn', 'precision', 'recall', 'f1']}
        return cm, stats, {}
    
    gt_labels_np = np.array(gt_labels)
    pred_labels_np = np.array(pred_labels)
    pred_scores_np = np.array(pred_scores)

    matched_gt = np.full(len(gt_labels_np), False)

    sorted_indices = np.argsort(-pred_scores_np)
    pred_indices = sorted_indices[pred_scores_np[sorted_indices] >= conf_threshold]

    for pred_idx in pred_indices:
        if iou_matrix.shape[0] == 0:
            best_iou = 0.0
            best_gt_idx = -1
        else:
            best_gt_idx = np.argmax(iou_matrix[:, pred_idx])
            best_iou = iou_matrix[best_gt_idx, pred_idx]
        
        pred_cls = pred_labels_np[pred_idx]
        pred_cls_idx = class_to_idx[pred_cls]

        if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
            gt_cls = gt_labels_np[best_gt_idx]
            gt_cls_idx = class_to_idx[gt_cls]
            cm[pred_cls_idx, gt_cls_idx] += 1
            matched_gt[best_gt_idx] = True
        else:
            cm[pred_cls_idx, -1] += 1  # pred to background

    unmatched_gt = np.where(~matched_gt)[0]
    for idx in unmatched_gt:
        gt_cls = gt_labels_np[idx]
        gt_cls_idx = class_to_idx[gt_cls]
        cm[-1, gt_cls_idx] += 1  # background to gt

    tp = np.diag(cm[:num_classes, :num_classes])
    fp = cm[:num_classes, -1]
    fn = cm[-1, :num_classes]
    
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall, 
        out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0
    )
 
    class_stats = {
        cls: {
            'tp': int(tp[idx]),
            'fp': int(fp[idx]),
            'fn': int(fn[idx]),
            'precision': round(precision[idx].item(), 3),
            'recall': round(recall[idx].item(), 3),
            'f1': round(f1[idx].item(), 3),
        }
        for cls, idx in class_to_idx.items()
    }

    stats = {
        'tp': int(np.sum(tp)),
        'fp': int(np.sum(fp)),
        'fn': int(np.sum(fn)),
        'precision': round(np.mean(precision).item(), 3),
        'recall': round(np.mean(recall).item(), 3),
        'f1': round(np.mean(f1).item(), 3),
    }

    return cm, stats, class_stats

def compute_map(
    gt: List[BBox], 
    preds: List[PredBBox], 
    iou_thresholds: List[float]
) -> float:
    if len(gt) == 0:
        gt_boxes = gt_labels = np.array([])
    else:
        gt_boxes = gt[:,1:]
        gt_labels = gt[:,0]
    if len(preds) == 0:
        pred_boxes = pred_labels = pred_scores = np.array([])
    else:
        pred_boxes = preds[:,0:4]
        pred_labels = preds[:,5]
        pred_scores = preds[:,4]
    iou = compute_iou_matrix(gt_boxes, pred_boxes)
    
    aps = {}
    for iou_thresh in iou_thresholds:
        cm, stats, class_stats = compute_confusion_matrix(
            gt_labels, 
            pred_labels, 
            pred_scores, 
            iou,
            iou_threshold=iou_thresh,
            conf_threshold=0.)
        for k, v in class_stats.items():
            if k not in aps:
                aps.update({k:[]})
            aps[k].append(v['precision'])
    for k in aps:
        aps[k] = round(sum(aps[k])/len(aps[k]), 3)
    return aps, class_stats

def compute_metrics(lbls, preds):
    aps, class_stats = compute_map(lbls, preds, iou_thresholds=[x / 100 for x in range(50, 100, 5)])
    map50_95 = aps.copy()
    aps, class_stats = compute_map(lbls, preds, iou_thresholds=[0.5])
    for k in aps:
        class_stats[k]['map50'] = aps[k]
        class_stats[k]['map50_95'] = map50_95[k]
    return class_stats

def sum_metrics(metrics):
    classes = {}
    for m in metrics:
        for k, v in m.items():
            if k not in classes:
                classes.update({k:[{}, 0]})
            classes[k][1] += 1
            for name in v:
                if name not in classes[k][0]:
                    classes[k][0].update({name:0})
                classes[k][0][name] += v[name]
    # for k, v in classes.items():
    #     ms, l = v
    #     for m in ms:
    #         classes[k][0][m] = round(classes[k][0][m] / l, 3)
    for k in classes:
        classes[k] = classes[k][0]
    return classes

# ======== <-- Calculate metrics ==========================

if __name__ == "__main__":
    data = "../../../../data/valid/roofs/"
    model = "crossroads_roofs-yolov8m-1.2.3.4.9_12(best).pt"
    metrics = validation(data, model)
    print("\tP\tR\tf1\tmap50\tmap50_95")
    for m, v in metrics.items():
        print("{}:\t{}\t{}\t{}\t{}\t{}".format(
            m,
            round(v['precision'], 3),
            round(v["recall"], 3),
            round(v["f1"], 3),
            round(v["map50"], 3),
            round(v["map50_95"], 3)
        ))
    
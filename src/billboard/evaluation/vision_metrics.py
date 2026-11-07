from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from billboard.config import CLASSES
from billboard.evaluation.cvat_video import VehicleBox


def xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def detection_metrics(frames: list[tuple[int, list[VehicleBox], list[VehicleBox]]]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = {name: key for key, name in CLASSES.items()}
    images, labels, detections = [], [], []
    for frame, truth, predictions in frames:
        images.append({"id": frame})
        for box in truth:
            rect = xywh(box.box)
            labels.append({"id": len(labels) + 1, "image_id": frame, "category_id": names[box.class_name],
                           "bbox": rect, "area": rect[2] * rect[3], "iscrowd": 0})
        for box in predictions:
            detections.append({"image_id": frame, "category_id": names[box.class_name],
                               "bbox": xywh(box.box), "score": box.confidence})
    if not labels:
        raise ValueError("The evaluation frames contain no manually labelled vehicles")
    seen = {label["category_id"] for label in labels}
    with redirect_stdout(StringIO()):
        ground = COCO()
        ground.dataset = {"info": {}, "images": images, "annotations": labels,
                          "categories": [{"id": key, "name": name} for key, name in CLASSES.items()]}
        ground.createIndex()
        if detections:
            predicted = ground.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {**ground.dataset, "annotations": []}
            predicted.createIndex()
        evaluator = COCOeval(ground, predicted, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    by_class = {}
    for index, category in enumerate(evaluator.params.catIds):
        values = evaluator.eval["precision"][0, :, index, 0, -1]
        by_class[CLASSES[category]] = float(values[values >= 0].mean()) if category in seen else None
    return {"map_50": float(evaluator.stats[1]), "map_50_95": float(evaluator.stats[0]),
            "ap_50_by_class": by_class, "manual_boxes": len(labels), "predicted_boxes": len(detections)}


def box_iou(left: list[VehicleBox], right: list[VehicleBox]) -> np.ndarray:
    a = np.asarray([box.box for box in left], dtype=float).reshape(-1, 4)
    b = np.asarray([box.box for box in right], dtype=float).reshape(-1, 4)
    overlap = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0)
    intersection = overlap.prod(axis=2)
    union = (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None] + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def tracking_metrics(frames: list[tuple[int, list[VehicleBox], list[VehicleBox]]]) -> dict:
    from trackeval.metrics import CLEAR, HOTA, Identity

    by_class = {}
    for name in CLASSES.values():
        gt_ids, pred_ids = {}, {}
        sequence = {"gt_ids": [], "tracker_ids": [], "similarity_scores": []}
        for _, truth, predictions in frames:
            truth = [item for item in truth if item.class_name == name]
            predictions = [item for item in predictions if item.class_name == name]
            for items, mapping, field in ((truth, gt_ids, "gt_ids"), (predictions, pred_ids, "tracker_ids")):
                ids = [mapping.setdefault(item.track_id, len(mapping)) for item in items]
                if len(set(ids)) != len(ids):
                    raise ValueError("A tracking ID appears more than once on one frame")
                sequence[field].append(np.asarray(ids, dtype=int))
            sequence["similarity_scores"].append(box_iou(truth, predictions))
        num_gt = sum(map(len, sequence["gt_ids"]))
        num_pred = sum(map(len, sequence["tracker_ids"]))
        if not num_gt:
            by_class[name] = {"manual_boxes": 0, "predicted_boxes": num_pred, "hota": None, "mota": None, "idf1": None}
            continue
        sequence.update(num_gt_ids=len(gt_ids), num_tracker_ids=len(pred_ids), num_timesteps=len(frames),
                        num_gt_dets=num_gt, num_tracker_dets=num_pred)
        options = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
        clear = CLEAR(options).eval_sequence(sequence)
        identity = Identity(options).eval_sequence(sequence)
        hota = HOTA().eval_sequence(sequence)
        by_class[name] = {"manual_boxes": num_gt, "predicted_boxes": num_pred, "hota": float(np.mean(hota["HOTA"])),
                          "mota": float(clear["MOTA"]), "idf1": float(identity["IDF1"]),
                          "id_switches": int(clear["IDSW"]), "misses": int(clear["CLR_FN"]),
                          "false_positives": int(clear["CLR_FP"])}
    if not any(row["manual_boxes"] for row in by_class.values()):
        raise ValueError("The evaluation frames contain no manually labelled vehicles")
    return {"by_class": by_class}

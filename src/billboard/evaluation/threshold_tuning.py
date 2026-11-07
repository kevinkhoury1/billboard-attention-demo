import numpy as np

from billboard.config import CLASSES
from billboard.evaluation.vision_metrics import box_iou


def tune_thresholds(frames: list, minimum_iou: float = .5) -> dict:
    """Find the confidence that maximizes box F1 for each class on development footage."""
    result = {}
    for name in CLASSES.values():
        scored = []
        truth_count = 0
        for _, truth, predictions in frames:
            truth = [box for box in truth if box.class_name == name]
            predictions = sorted((box for box in predictions if box.class_name == name),
                                 key=lambda box: box.confidence, reverse=True)
            truth_count += len(truth)
            unmatched = set(range(len(truth)))
            for box, overlaps in zip(predictions, box_iou(predictions, truth), strict=True):
                best = max(unmatched, key=lambda index: overlaps[index], default=None)
                match = best is not None and overlaps[best] >= minimum_iou
                if match:
                    unmatched.remove(best)
                scored.append((box.confidence, match))
        if not truth_count or not any(match for _, match in scored):
            result[name] = {"threshold": None, "manual_boxes": truth_count}
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        scores = np.asarray([score for score, _ in scored])
        last_equal_score = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
        true_positives = np.cumsum([match for _, match in scored])[last_equal_score]
        counts = last_equal_score + 1
        f1 = 2 * true_positives / (counts + truth_count)
        best = int(f1.argmax())
        result[name] = {
            "threshold": float(scores[last_equal_score[best]]), "manual_boxes": truth_count,
            "precision": float(true_positives[best] / counts[best]),
            "recall": float(true_positives[best] / truth_count), "f1": float(f1[best]),
        }
    return result

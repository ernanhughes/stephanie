# stephanie/components/nexus/vpm/candidate_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from stephanie.components.nexus.utils.visual_thought import (VisualThoughtOp,
                                                             VisualThoughtType)
from stephanie.components.nexus.vpm.state_machine import Thought


def _nms(boxes: List[Tuple[int,int,int,int]], scores: List[float], iou_thresh=0.3):
    keep = []
    x1 = np.array([b[0] for b in boxes]); y1 = np.array([b[1] for b in boxes])
    x2 = np.array([b[2] for b in boxes]); y2 = np.array([b[3] for b in boxes])
    areas = (x2-x1) * (y2-y1)
    order = np.argsort(scores)[::-1]
    while len(order):
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2-xx1, 0, None); h = np.clip(yy2-yy1, 0, None)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep

@dataclass
class CandidatePolicy:
    patch: int = 64
    zoom_scale_min: float = 1.5
    zoom_scale_max: float = 3.0
    topk: int = 5
    iou_thresh: float = 0.3

    def propose(self, interesting: np.ndarray, H: int, W: int) -> List[Thought]:
        k = max(8, min(self.patch, min(H, W)))
        # scan grid
        boxes, scores = [], []
        for y in range(0, H-k+1, max(1, k//2)):
            for x in range(0, W-k+1, max(1, k//2)):
                m = float(interesting[y:y+k, x:x+k].mean())
                boxes.append((x, y, x+k, y+k)); scores.append(m)
        if not boxes:
            cx, cy = W//2, H//2
            return [Thought("zoom_center", [VisualThoughtOp(VisualThoughtType.ZOOM, {"center": (cx, cy), "scale": 2.0})], cost=1.0)]

        # NMS + topk
        keep = _nms(boxes, scores, self.iou_thresh)[: self.topk]
        keep_boxes = [boxes[i] for i in keep]; keep_scores = [scores[i] for i in keep]
        # build thoughts
        thoughts: List[Thought] = []
        for (x1,y1,x2,y2), s in sorted(zip(keep_boxes, keep_scores), key=lambda t: t[1], reverse=True):
            cx, cy = (x1+x2)//2, (y1+y2)//2
            thoughts.append(
                Thought(
                    name="zoom_focus",
                    ops=[VisualThoughtOp(VisualThoughtType.ZOOM, {
                        "center": (cx, cy),
                        "scale": float(np.random.uniform(self.zoom_scale_min, self.zoom_scale_max))
                    })],
                    cost=1.0
                )
            )
            thoughts.append(
                Thought(
                    name="bbox_emphasize",
                    ops=[VisualThoughtOp(VisualThoughtType.BBOX, {
                        "xyxy": (x1, y1, x2, y2),
                        "width": 2, "intensity": 0.8
                    })],
                    cost=0.3
                )
            )
        return thoughts

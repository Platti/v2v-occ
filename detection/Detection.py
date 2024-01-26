from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from util.Box import Box


@dataclass
class Detection:
    """Dataclass for detections"""
    box: Box
    confidence: float
    tll: Box = None
    tlr: Box = None
    eq_threshold: ClassVar[float] = 0.5
    ioa_threshold: ClassVar[float] = 0.9

    @property
    def valid(self) -> bool:
        """check if detection has a valid aspect ratio for a car"""
        return 1 < self.box.aspect_ratio < 4 / 3

    def __str__(self):
        return f"Detection({self.box}, confidence: {self.confidence * 100:.1f}%)"

    def __eq__(self, other):
        if isinstance(other, Detection):
            return self.box.calc_iou(other.box) > Detection.eq_threshold
        return False

    @staticmethod
    def clean_duplicates(detections: [Detection]) -> [Detection]:
        """remove duplicate detections, that are either very similar or mainly overlapped by other detections"""
        detections.sort(key=lambda d: d.box.area, reverse=True)
        detections = [d1 for i, d1 in enumerate(detections)
                      if not any((d1.box.calc_intersection_over_area(d2.box) > Detection.ioa_threshold
                                  for d2 in detections[:i]))]

        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = [d1 for i, d1 in enumerate(detections) if not any((d1 == d2 for d2 in detections[:i]))]

        return detections

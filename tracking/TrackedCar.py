import itertools
import cv2
from dataclasses import dataclass, field
from typing import ClassVar
from detection.Detection import Detection

"""Generator for unique incremental IDs"""
id_gen = itertools.count()


@dataclass
class TrackedCar:
    """Dataclass for a tracked car"""
    id: int = field(init=False)
    start_frame: int
    detection: Detection
    tracker: cv2.Tracker
    age: int = 0
    receiving: bool = False

    """Threshold for checking if Detection matches another one for tracking"""
    iou_match_threshold: ClassVar[float] = 0.3
    """Threshold for checking if a Detection is mainly overlapped by another on for tracking"""
    overlap_match_threshold: ClassVar[float] = 0.9
    """Minimum age in number of frames of car to be relevant"""
    relevant_age: ClassVar[int] = 30

    def __post_init__(self):
        self.id = next(id_gen)

    def matches(self, detection: Detection) -> bool:
        """check if a detection matches the stored detection for tracking"""
        high_iou = self.detection.box.calc_iou(detection.box) > TrackedCar.iou_match_threshold
        high_overlap = self.detection.box.calc_overlapping_rate(detection.box) > TrackedCar.overlap_match_threshold
        return high_iou or high_overlap

    @property
    def relevant(self) -> bool:
        return self.age >= TrackedCar.relevant_age

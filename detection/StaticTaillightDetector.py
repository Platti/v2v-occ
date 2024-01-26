import numpy as np

from detection.Detection import Detection
from detection.TaillightDetector import TaillightDetector
from util.Box import Box


class StaticTaillightDetector(TaillightDetector):
    def detect_taillights(self, img: np.ndarray, car: Detection = None) -> (Detection, Detection):
        assert (car is not None, "StaticTaillightDetector depends on detection of car which was None!")
        tl_top = car.box.top + car.box.height // 4
        tl_bottom = car.box.top + 7 * car.box.height // 12

        tll_left = car.box.left
        tll_right = car.box.left + 5 * car.box.width // 16

        tlr_left = car.box.right - 5 * car.box.width // 16
        tlr_right = car.box.right

        tll = Detection(Box(tll_left, tl_top, tll_right, tl_bottom), confidence=1.0)
        tlr = Detection(Box(tlr_left, tl_top, tlr_right, tl_bottom), confidence=1.0)

        return tll, tlr

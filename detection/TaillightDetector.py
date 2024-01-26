from abc import ABC, abstractmethod
import numpy as np
from detection.Detection import Detection


class TaillightDetector(ABC):
    """Abstract Base Class for taillight detector implementations"""

    @abstractmethod
    def detect_taillights(self, img: np.ndarray, car: Detection = None) -> [Detection]:
        """
        Detect the taillights of a car in an image

        img: whole image
        car: detection of car (only necessary for daylight)
        """
        pass

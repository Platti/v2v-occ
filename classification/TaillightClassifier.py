import numpy as np
from abc import ABC, abstractmethod


class TaillightClassifier(ABC):
    """Abstract Base Class for taillight state classifiers"""

    @abstractmethod
    def classify_taillight(self, img: np.ndarray, last_img: np.ndarray = None, tl_id: int = 0) -> bool:
        """classify state of single taillight image"""
        pass

    @abstractmethod
    def classify_taillight_batch(self, imgs: [np.array], last_imgs: [np.array] = None) -> [bool]:
        """classify states of multiple taillight in image batch"""
        pass

from abc import ABC, abstractmethod
import numpy as np
from detection.Detection import Detection


class CarDetector(ABC):
    """Abstract Base Class for car detector implementations"""

    @abstractmethod
    def detect_cars(self, img: np.ndarray) -> [Detection]:
        """Detect all cars in an image"""
        pass

    @abstractmethod
    def detect_cars_batch(self, imgs: [np.ndarray]) -> [[Detection]]:
        """Detect all cars in a batch of images"""
        pass

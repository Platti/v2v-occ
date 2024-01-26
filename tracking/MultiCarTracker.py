from abc import ABC, abstractmethod

import numpy as np

from tracking.TrackedCar import TrackedCar
from util.Observer import TrackerObserver


class MultiCarTracker(ABC):
    """Abstract Base Class for multi-car trackers"""

    def __init__(self):
        self.observers = []

    @abstractmethod
    def feed_img(self, img) -> None:
        """feed next image to the tracker"""
        pass

    @abstractmethod
    def run_tracking(self) -> None:
        """run the tracking on the currently fed image(s)"""
        pass

    @abstractmethod
    def get_tracked_cars(self) -> [TrackedCar]:
        """get list of tracked cars with their current position"""
        pass

    @abstractmethod
    def get_tracked_cars_history(self) -> [TrackedCar]:
        """get all tracked cars as history including lost cars"""
        pass

    @abstractmethod
    def get_current_img(self) -> [np.ndarray]:
        """get currently processed image"""
        pass

    @abstractmethod
    def get_last_img(self) -> [np.ndarray]:
        """get previously processed image"""
        pass

    def add_observer(self, observer: TrackerObserver):
        self.observers.append(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.update()

    def get_tracking_delay(self) -> int:
        """get tracking delay in number of frames, e.g., due to using the center of a sliding window."""
        return 0

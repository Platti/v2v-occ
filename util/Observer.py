from __future__ import annotations
from abc import ABC, abstractmethod

import tracking.MultiCarTracker as Tracking


class TrackerObserver(ABC):
    """Observer class for trackers"""

    def __init__(self, tracker: Tracking.MultiCarTracker):
        self.tracker = tracker
        self.tracker.add_observer(self)

    @abstractmethod
    def update(self) -> None:
        """read and react to the information from the self.tracker object"""
        pass

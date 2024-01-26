import logging

import cv2
import numpy as np

from detection.CarDetector import CarDetector
from detection.Detection import Detection
from tracking.SimpleMultiCarTracker import SimpleMultiCarTracker
from tracking.TrackedCar import TrackedCar
from util.Box import Box
from util.Util import pairwise


class DifferentialMultiCarTracker(SimpleMultiCarTracker):
    """
    Extension for SimpleMultiCarTracker to not only use the images as is but add support to use differential brightness images

    last_frames: list of queued images in grayscale
    """
    last_frames: [np.ndarray]
    last_frames_bgr: [np.ndarray]
    diff_frame: np.ndarray
    diff_frame_updated: bool
    dark_mode: bool

    def __init__(self, detector: CarDetector, num_frames: int):
        super().__init__(detector)
        self.last_frames = []
        self.last_frames_bgr = []
        self.num_frames = num_frames
        self.diff_frame_updated = False
        self.dark_mode = False

    def feed_img(self, img) -> None:
        self.last_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        self.last_frames_bgr.append(img)
        super().feed_img(self.last_frames_bgr[len(self.last_frames_bgr) // 2])
        #super().feed_img(self.last_frames_bgr[0])
        if len(self.last_frames) > self.num_frames:
            self.last_frames.pop(0)
            self.last_frames_bgr.pop(0)
        self.diff_frame_updated = False
        self.adjust_refresh_cycle()

    def adjust_refresh_cycle(self) -> None:
        if np.mean(self.current_img) < 30:
            self.dark_mode = True
            self.refresh_cycle = 5
        else:
            self.dark_mode = False
            self.refresh_cycle = 20

    def run_new_detection(self) -> [Detection]:
        return self.detector.detect_cars(self.get_differential_frame())

    def get_differential_frame(self):
        if not self.diff_frame_updated:
            diff_frame = np.zeros_like(self.last_frames[0], dtype=float)
            for frame1, frame2 in pairwise(self.last_frames):
                diff_frame = diff_frame + cv2.absdiff(frame1, frame2)
            self.diff_frame = diff_frame / np.max(diff_frame)
            self.diff_frame_updated = True
        return self.diff_frame

    def update_using_trackers(self):
        if not self.dark_mode:
            super().update_using_trackers()

    def get_tracking_delay(self) -> int:
        return self.num_frames // 2

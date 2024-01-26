from abc import ABC

import cv2
import numpy as np

from classification.TaillightClassifier import TaillightClassifier
from detection.TaillightDetector import TaillightDetector
from tracking.MultiCarTracker import MultiCarTracker
from util import Util
from util.Observer import TrackerObserver


class Visualizer(TrackerObserver, ABC):
    """Abstract Base Class for visualizers for car tracking"""

    def __init__(self,
                 tracker: MultiCarTracker,
                 tl_detector: TaillightDetector = None,
                 tl_classifier: TaillightClassifier = None):
        super().__init__(tracker)
        self.tl_detector = tl_detector
        self.tl_classifier = tl_classifier
        self.color = (0, 255, 0)
        self.color_text = (0, 0, 0)
        self.color_ready = (0, 200, 200)
        self.color_ignore = (150, 150, 150)
        self.timestamps = [Util.current_time_ms() - 1]

    def tick(self) -> None:
        """timing tick used to record timestamps for FPS calculation"""
        self.timestamps.append(Util.current_time_ms())
        self.timestamps = self.timestamps[-100:]

    def draw_tracked_cars(self, img) -> np.ndarray:
        """draw bounding boxes of tracked cars on image including ID and confidence"""
        for car in self.tracker.get_tracked_cars():
            if car.receiving:
                color = self.color
            elif car.relevant:
                color = self.color_ready
            else:
                color = self.color_ignore

            img = self.draw_car_on(img, car.id, car.detection, color, self.color_text)
        return img

    @staticmethod
    def draw_car_on(img, car_id, detection, color, color_text):
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (detection.box.left, detection.box.top),
            (detection.box.right, detection.box.bottom),
            color=color,
            thickness=2)
        """
        if detection.tll is not None:
            cv2.rectangle(
                overlay,
                (detection.tll.left, detection.tll.top),
                (detection.tll.right, detection.tll.bottom),
                color=(255, 0, 0),
                thickness=1)
        if detection.tlr is not None:
            cv2.rectangle(
                overlay,
                (detection.tlr.left, detection.tlr.top),
                (detection.tlr.right, detection.tlr.bottom),
                color=(0, 0, 255),
                thickness=1)
        """
        cv2.rectangle(
            overlay,
            (detection.box.left, detection.box.top),
            (detection.box.left + 100, detection.box.top - 30),
            color=color,
            thickness=cv2.FILLED)
        cv2.putText(
            overlay,
            f'conf: {int(detection.confidence * 100)}',
            (detection.box.left, detection.box.top - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color_text)
        cv2.putText(
            overlay,
            f'id: {car_id}',
            (detection.box.left, detection.box.top - 18),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color_text)
        img = cv2.addWeighted(
            src1=overlay,
            alpha=detection.confidence,
            src2=img,
            beta=1 - detection.confidence,
            gamma=0)
        return img

    def calc_fps(self) -> int:
        """calculate current FPS"""
        return len(self.timestamps) * 1000 // (self.timestamps[-1] - self.timestamps[0])

    def draw_fps(self, img) -> np.ndarray:
        """draw current FPS in top-left corner"""
        cv2.rectangle(
            img,
            (0, 0),
            (80, 20),
            color=self.color,
            thickness=cv2.FILLED)
        cv2.putText(
            img,
            f'FPS: {self.calc_fps()}',
            (2, 16),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=self.color_text)
        return img

    @staticmethod
    def draw_frame_id(img, frame_id) -> np.ndarray:
        cv2.rectangle(
            img,
            (0, 0),
            (120, 20),
            color=(0, 0, 0),
            thickness=cv2.FILLED)
        cv2.putText(
            img,
            f'Frame: {frame_id}',
            (2, 16),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255))
        return img

    def draw_taillights(self, canvas_img) -> np.ndarray:
        """draw bounding boxes of taillights, thickness shows state"""
        car = next((car for car in self.tracker.get_tracked_cars() if car.relevant), None)
        if car and self.tl_detector and self.tl_classifier:
            car.receiving = True
            img = self.tracker.get_current_img()

            tll, tlr = self.tl_detector.detect_taillights(img, car.detection)
            state_tll, state_tlr = self.tl_classifier.classify_taillight_batch([tll.box.crop_from(img),
                                                                                tlr.box.crop_from(img)])

            cv2.rectangle(
                canvas_img,
                (tll.box.left, tll.box.top),
                (tll.box.right, tll.box.bottom),
                color=(255, 0, 0),
                thickness=2 if state_tll else 1)
            cv2.rectangle(
                canvas_img,
                (tlr.box.left, tlr.box.top),
                (tlr.box.right, tlr.box.bottom),
                color=(0, 0, 255),
                thickness=2 if state_tlr else 1)

        return canvas_img

    @staticmethod
    def draw_taillights_on(img, tll_detection, tlr_detection, tll_state, tlr_state):
        cv2.rectangle(
            img,
            (tll_detection.box.left, tll_detection.box.top),
            (tll_detection.box.right, tll_detection.box.bottom),
            color=(255, 0, 0),
            thickness=2 if tll_state else 1)
        cv2.rectangle(
            img,
            (tlr_detection.box.left, tlr_detection.box.top),
            (tlr_detection.box.right, tlr_detection.box.bottom),
            color=(0, 0, 255),
            thickness=2 if tlr_state else 1)
        return img

    @staticmethod
    def draw_taillights_transition_on(img, tll_detection, tlr_detection, tll_state, tlr_state):
        cv2.line(
            img,
            (tll_detection.box.left, tll_detection.box.top),
            (tll_detection.box.right, tll_detection.box.top),
            color=(255, 0, 0),
            thickness=2 if tll_state else 1)
        cv2.line(
            img,
            (tll_detection.box.right, tll_detection.box.top),
            (tll_detection.box.right, tll_detection.box.bottom),
            color=(255, 0, 0),
            thickness=1 if tll_state else 2)
        cv2.line(
            img,
            (tll_detection.box.left, tll_detection.box.bottom),
            (tll_detection.box.right, tll_detection.box.bottom),
            color=(255, 0, 0),
            thickness=2 if tll_state else 1)
        cv2.line(
            img,
            (tll_detection.box.left, tll_detection.box.top),
            (tll_detection.box.left, tll_detection.box.bottom),
            color=(255, 0, 0),
            thickness=1 if tll_state else 2)

        cv2.line(
            img,
            (tlr_detection.box.left, tlr_detection.box.top),
            (tlr_detection.box.right, tlr_detection.box.top),
            color=(0, 0, 255),
            thickness=2 if tlr_state else 1)
        cv2.line(
            img,
            (tlr_detection.box.right, tlr_detection.box.top),
            (tlr_detection.box.right, tlr_detection.box.bottom),
            color=(0, 0, 255),
            thickness=1 if tlr_state else 2)
        cv2.line(
            img,
            (tlr_detection.box.left, tlr_detection.box.bottom),
            (tlr_detection.box.right, tlr_detection.box.bottom),
            color=(0, 0, 255),
            thickness=2 if tlr_state else 1)
        cv2.line(
            img,
            (tlr_detection.box.left, tlr_detection.box.top),
            (tlr_detection.box.left, tlr_detection.box.bottom),
            color=(0, 0, 255),
            thickness=1 if tlr_state else 2)

        return img

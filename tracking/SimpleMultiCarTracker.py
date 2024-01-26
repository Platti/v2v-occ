import cv2
import numpy as np

from detection.Detection import Detection
from tracking.MultiCarTracker import MultiCarTracker
from tracking.TrackedCar import TrackedCar
from detection.CarDetector import CarDetector
from util.Box import Box


class SimpleMultiCarTracker(MultiCarTracker):
    """A simple implementation of a multi-car tracker using a given car detector
    and a MOSSE tracker for the frames between detections

    refresh_cycle: number of frames between two detections
    """

    frame_id: int = 0
    refresh_cycle: int = 10
    detector: CarDetector
    tracked_cars: [TrackedCar]
    tracked_cars_lost: [TrackedCar]
    current_img: np.ndarray
    last_img: np.ndarray

    def __init__(self, detector: CarDetector):
        super().__init__()
        self.detector = detector
        self.tracked_cars = []
        self.tracked_cars_lost = []
        self.current_img = None

    def feed_img(self, img) -> None:
        self.last_img = self.current_img
        self.current_img = img

    def run_tracking(self) -> None:
        self.cleanup_tracked_cars()

        if self.frame_id % self.refresh_cycle == 0:
            detections = self.run_new_detection()
            self.merge_detections_and_trackers(detections)
        else:
            self.update_using_trackers()

        self.notify_observers()
        self.frame_id += 1

    def run_new_detection(self) -> [Detection]:
        return self.detector.detect_cars(self.current_img)

    def cleanup_tracked_cars(self) -> None:
        """remove tracked cars with a confidence less than 0 and increase age"""
        for tracked_car in self.tracked_cars:
            tracked_car.age += 1
            if tracked_car.detection.confidence <= 0:
                self.tracked_cars_lost.append(tracked_car)
                self.tracked_cars.remove(tracked_car)

    def merge_detections_and_trackers(self, car_detections, debug=False):
        """associate new detections with previously tracked cars and update them or store them as new detection"""
        debug_img = self.current_img.copy()
        debug_check = [0, 0, 0, 0]

        for tracked_car in self.tracked_cars:
            tracked_car.detection.confidence -= 0.05
            if debug:
                SimpleMultiCarTracker.draw_debug_rect_on(debug_img, tracked_car.detection, (208, 224, 64), 1)

        for detection in car_detections:
            if not detection.valid:
                if debug:
                    SimpleMultiCarTracker.draw_debug_rect_on(debug_img, detection, (200, 200, 200), 1)
                    # debug_check[0] = 1
                continue

            for tracked_car in self.tracked_cars:
                if tracked_car.matches(detection):
                    if debug:
                        SimpleMultiCarTracker.draw_debug_rect_on(debug_img, detection, (0, 165, 255))
                        SimpleMultiCarTracker.draw_debug_rect_on(debug_img, tracked_car.detection, (208, 224, 64))
                    if detection.confidence > tracked_car.detection.confidence:
                        if debug:
                            SimpleMultiCarTracker.draw_debug_rect_on(debug_img, detection, (0, 165, 255),
                                                                     text=f"IOU: {detection.box.calc_iou(tracked_car.detection.box) * 100:.0f}%")
                            debug_check[1] = 1
                        self.refresh_detection(tracked_car, detection)
                    elif debug:
                        SimpleMultiCarTracker.draw_debug_rect_on(debug_img, tracked_car.detection, (208, 224, 64),
                                                                 text=f"IOU: {detection.box.calc_iou(tracked_car.detection.box) * 100:.0f}%")
                        debug_check[2] = 1
                    break
            else:
                if debug:
                    SimpleMultiCarTracker.draw_debug_rect_on(debug_img, detection, (0, 255, 255))
                    debug_check[3] = 1
                self.tracked_cars.append(self.create_tracked_car(detection))

        if debug:
            cv2.imshow("tracking debug", debug_img)
            if sum(debug_check) >= 3:
                cv2.imwrite("output/testing/car_tracking/bright/debug.png", debug_img)
                self.notify_observers()
                cv2.waitKey(0)

    @staticmethod
    def draw_debug_rect_on(img, detection, color, thickness=2, text=None):
        cv2.rectangle(
            img,
            (detection.box.left, detection.box.top),
            (detection.box.right, detection.box.bottom),
            color=color,
            thickness=thickness)
        if text:
            cv2.rectangle(
                img,
                (detection.box.left + 3, detection.box.bottom - 3),
                (detection.box.left + 80, detection.box.bottom - 17),
                color=color,
                thickness=cv2.FILLED)
            cv2.putText(
                img,
                text,
                (detection.box.left + 5, detection.box.bottom - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0))

    def create_tracked_car(self, detection):
        """create a new tracked car"""
        return TrackedCar(start_frame=self.frame_id,
                          detection=detection,
                          tracker=SimpleMultiCarTracker.create_tracker(self.current_img, detection))

    def refresh_detection(self, tracked_car: TrackedCar, detection) -> None:
        """refresh an old detection with the associated new one"""
        tracked_car.detection = detection
        tracked_car.tracker = SimpleMultiCarTracker.create_tracker(self.current_img, detection)

    @staticmethod
    def create_tracker(img, detection) -> cv2.Tracker:
        """create a new MOSSE tracker and initialize it"""
        tracker = cv2.legacy_TrackerMOSSE.create()
        tracker.init(img, detection.box.as_ltwh())
        return tracker

    def update_using_trackers(self):
        """update the tracked cars using the MOSSE trackers and
        adjust the confidence depending on the tracking result"""
        for tracked_car in self.tracked_cars:
            success, box_ltwh = tracked_car.tracker.update(self.current_img)
            if success:
                tracked_car.detection.box = Box.from_ltwh(box_ltwh)
                tracked_car.detection.confidence -= 0.002
            else:
                tracked_car.detection.confidence -= 0.005

    def get_tracked_cars(self) -> [TrackedCar]:
        return self.tracked_cars.copy()

    def get_tracked_cars_history(self) -> [TrackedCar]:
        history = []
        history.extend(self.tracked_cars_lost.copy())
        history.extend(self.tracked_cars.copy())
        history.sort(key=lambda car: car.id)
        return history

    def get_current_img(self) -> [np.ndarray]:
        return self.current_img.copy()

    def get_last_img(self) -> [np.ndarray]:
        return self.last_img.copy() if self.last_img is not None else None

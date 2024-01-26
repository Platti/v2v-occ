from detection.CarDetector import CarDetector
from detection.Detection import Detection
from tracking.DifferentialMultiCarTracker import DifferentialMultiCarTracker


class DualMultiCarTracker(DifferentialMultiCarTracker):

    bright_detector: CarDetector

    def __init__(self, bright_detector: CarDetector, dark_detector: CarDetector, num_frames: int):
        super().__init__(detector=dark_detector, num_frames=num_frames)
        self.bright_detector = bright_detector

    def run_new_detection(self) -> [Detection]:
        if not self.dark_mode:
            detections = self.bright_detector.detect_cars(self.current_img)
            if len(detections) > 0:
                return detections

        return super().run_new_detection()


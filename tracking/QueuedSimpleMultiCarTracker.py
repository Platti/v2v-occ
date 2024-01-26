from detection.CarDetector import CarDetector
from tracking.SimpleMultiCarTracker import SimpleMultiCarTracker
import numpy as np


class QueuedSimpleMultiCarTracker(SimpleMultiCarTracker):
    """
    Extension for SimpleMultiCarTracker to not queue the fed images and track them batch-wise

    img_queue: list of queued images
    head: frame index of first element in image queue
    """
    img_queue: [np.ndarray]
    head: int

    def __init__(self, detector: CarDetector):
        super().__init__(detector)
        self.img_queue = []
        self.head = 0
        self.refresh_cycle = 10

    def feed_img(self, img) -> None:
        super().feed_img(img)
        self.img_queue.append(img)

    def run_tracking(self) -> None:
        start = self.head % self.refresh_cycle
        stop = len(self.img_queue)
        step = self.refresh_cycle
        detections_all = self.detector.detect_cars_batch(self.img_queue[start:stop:step])
        for i in range(stop):
            self.current_img = self.img_queue[i]
            self.cleanup_tracked_cars()
            if i in range(start, stop, step):
                self.merge_detections_and_trackers(detections_all[0])
                detections_all = detections_all[1:]
            else:
                self.update_using_trackers()
                pass
            self.notify_observers()
            self.frame_id += 1
        self.head += stop
        self.img_queue.clear()

import cv2

from classification.TaillightClassifier import TaillightClassifier
from detection.TaillightDetector import TaillightDetector
from tracking.MultiCarTracker import MultiCarTracker
from visualization.Visualizer import Visualizer


class OnScreenVisualizer(Visualizer):
    """Visualizer implementation that shows the result on screen"""

    def __init__(self, tracker: MultiCarTracker,
                 tl_detector: TaillightDetector = None,
                 tl_classifier: TaillightClassifier = None):
        super().__init__(tracker, tl_detector, tl_classifier)
        self.quit_requested = False

    def update(self) -> None:
        self.tick()
        img = self.tracker.get_current_img()
        img = self.draw_tracked_cars(img)
        img = self.draw_taillights(img)
        img = self.draw_fps(img)
        self.show_on_screen(img)

    def show_on_screen(self, img) -> None:
        cv2.imshow("tracking", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            self.quit_requested = True



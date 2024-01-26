import cv2

from classification.TaillightClassifier import TaillightClassifier
from detection.TaillightDetector import TaillightDetector
from tracking.MultiCarTracker import MultiCarTracker
from visualization.Visualizer import Visualizer


class VideoVisualizer(Visualizer):
    """Visualizer implementation that stores an mp4 video of the result"""

    def __init__(self, tracker: MultiCarTracker,
                 output_path: str,
                 tl_detector: TaillightDetector = None,
                 tl_classifier: TaillightClassifier = None):
        super().__init__(tracker, tl_detector, tl_classifier)
        self.output_path = output_path
        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.target_fps = 30

    def update(self) -> None:
        self.tick()
        img = self.tracker.get_current_img()
        img = self.draw_tracked_cars(img)
        img = self.draw_taillights(img)
        self.write_video(img)

    def write_video(self, img) -> None:
        if not self.video_writer:
            height, width, channels = img.shape
            self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.target_fps, (width, height))

        self.video_writer.write(img)

    def info(self):
        print(f"Writing video with speed: {self.calc_fps() / self.target_fps:.02f}x")

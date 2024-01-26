from pathlib import Path

import cv2

from classification.CNNTaillightClassifier import CNNTaillightClassifier
from detection.MobileNetSSDCarDetector import MobileNetSSDCarDetector
from detection.StaticTaillightDetector import StaticTaillightDetector
from detection.ZoomedDetector import ZoomedDetector
from detection.dark.DifferentialBrightnessDetector import DifferentialBrightnessDetector
from receiver.QueuedUDPSOOKReceiver import QueuedUDPSOOKReceiver
from receiver.UDPSOOKReceiver import UDPSOOKReceiver
from tracking.DualMultiCarTracker import DualMultiCarTracker
from tracking.QueuedSimpleMultiCarTracker import QueuedSimpleMultiCarTracker
from tracking.SimpleMultiCarTracker import SimpleMultiCarTracker
from util import Util
from visualization.OnScreenVisualizer import OnScreenVisualizer
from visualization.VideoVisualizer import VideoVisualizer


def main():
    input_path = "res/input/"
    filename = "VLC_sunny.mp4"

    cap = cv2.VideoCapture(f"{input_path}{filename}")
    bright_detector = ZoomedDetector(MobileNetSSDCarDetector(), ZoomedDetector.get_default_zoom_boxes())
    dark_detector = DifferentialBrightnessDetector()

    tracker = QueuedSimpleMultiCarTracker(bright_detector)

    visualizer = OnScreenVisualizer(tracker)

    receiver = QueuedUDPSOOKReceiver(tracker, StaticTaillightDetector(), CNNTaillightClassifier())
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret or visualizer.quit_requested:
            break

        frame = Util.crop_frame(frame)
        tracker.feed_img(frame)
        if frame_id % 99 == 0:
            tracker.run_tracking()

        frame_id += 1

    output_dir = "out/"

    output_path = f"{output_dir}{filename}_TS{Util.current_time_ms()}"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    receiver.store_data_overview(frame_id, output_path)
    receiver.store_data_files(output_path)


if __name__ == '__main__':
    main()

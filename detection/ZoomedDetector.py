from __future__ import annotations

import cv2
import numpy as np

from detection.CarDetector import CarDetector
from detection.Detection import Detection
from detection.ZoomBox import ZoomBox
from util.Box import Box


class ZoomedDetector(CarDetector):
    """Decorator class for CarDetectors to not just use the given image for detection
    but to use multiple zoomed images as batch"""

    def __init__(self, detector: CarDetector, zoom_boxes: [ZoomBox]):
        self.detector = detector
        self.zoom_boxes = zoom_boxes

    @staticmethod
    def get_default_zoom_boxes(img: np.ndarray = np.zeros((450, 800, 3))) -> [ZoomBox]:
        """get default zoom boxes: whole image plus zoom center/left/right"""
        height, width, channels = img.shape

        #return [
        #    ZoomBox(Box.from_ltwh([0, 0, width, height]), 1.0),
        #    ZoomBox(Box.from_ltwh([width // 2 - 200, height // 2 - 100, 300, 200]), 0.9),
        #    ZoomBox(Box.from_ltwh([width // 2 - 400, height // 2 - 100, 300, 200]), 0.8),
        #    ZoomBox(Box.from_ltwh([width // 2, height // 2 - 100, 300, 200]), 0.8),
        #]

        return [
            ZoomBox(Box.from_ltwh([0, 0, width, height]), 1.0),
            ZoomBox(Box.from_ltwh([width // 2 - 150, height // 2 - 150, 300, 200]), 0.9),
            ZoomBox(Box.from_ltwh([width // 2 - 350, height // 2 - 150, 300, 200]), 0.8),
            ZoomBox(Box.from_ltwh([width // 2 + 50, height // 2 - 150, 300, 200]), 0.8),
        ]

    def detect_cars(self, img: np.ndarray) -> [Detection]:
        imgs = [zoom_box.box.crop_from(img) for zoom_box in self.zoom_boxes]
        detections_zoomed_imgs = self.detector.detect_cars_batch(imgs)
        return self.merge_zoomed_detections(detections_zoomed_imgs)

    def detect_cars_batch(self, imgs: [np.ndarray]) -> [[Detection]]:
        zoom_imgs = [[] for _ in range(len(imgs))]
        for i, img in enumerate(imgs):
            zoom_imgs[i] = [zoom_box.box.crop_from(img) for zoom_box in self.zoom_boxes]
        zoom_imgs_flat = flatten(zoom_imgs)

        detections_flat = self.detector.detect_cars_batch(zoom_imgs_flat)
        detections_all = unflatten(detections_flat, len(self.zoom_boxes))
        return [self.merge_zoomed_detections(detections_zoomed_imgs) for detections_zoomed_imgs in detections_all]

    def merge_zoomed_detections(self, detections_zoomed_imgs):
        """adjust detections in the zoomed images in order to get the coordinates in the original image"""
        for detections_zoomed_img, zoom_box in zip(detections_zoomed_imgs, self.zoom_boxes):
            for detection in detections_zoomed_img:
                detection.box.reset_origin(x=zoom_box.box.left, y=zoom_box.box.top)
                detection.confidence *= zoom_box.weight
        detections_zoomed_flat = flatten(detections_zoomed_imgs)
        return Detection.clean_duplicates(detections_zoomed_flat)


def flatten(nested_list):
    """flatten a list of sub-lists"""
    return [item for sublist in nested_list for item in sublist]


def unflatten(flat_list, n):
    """create a list of sub-lists having n items per sub-list"""
    return [flat_list[i:i + n] for i in range(0, len(flat_list), n)]

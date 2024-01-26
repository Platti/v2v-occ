import logging

import numpy as np
import cv2

from skimage.feature import peak_local_max

from detection.CarDetector import CarDetector
from detection.Detection import Detection
from util.Box import Box


class DifferentialBrightnessDetector(CarDetector):

    def __init__(self, taillight_template_path="res/templates/taillight_template.png"):
        super().__init__()
        self.tl_template = cv2.imread(taillight_template_path, cv2.IMREAD_GRAYSCALE) / 255

    def detect_cars(self, img: np.ndarray) -> [Detection]:
        positions = peak_local_max(img, min_distance=25, threshold_abs=0.6)
        similarities = []
        for pos in positions:
            similarities.append(self.verify_taillight_position(pos, img))
        self.show_debug_image(img, positions, similarities)

        positions = [p for _, p in sorted(zip(similarities, positions), reverse=True)]

        detections = self.associate_taillights(positions, img)

        return detections

    def detect_cars_batch(self, imgs: [np.ndarray]) -> [[Detection]]:
        return [[]]

    def show_debug_image(self, img, positions, similarities):
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pos, sim in zip(positions, similarities):
            cv2.drawMarker(img, (pos[1], pos[0]), color=[0,255,255], markerType=cv2.MARKER_SQUARE, markerSize=50)
            cv2.putText(img, str(int(sim * 100)), (pos[1] - 22, pos[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,255])
        # cv2.imwrite("output/testing/car_tracking/dark/debug.png", img)
        cv2.imshow("detection image", img)

    def verify_taillight_position(self, pos, img, size=30):
        dl = size // 2
        img_part = img[pos[0] - dl:pos[0] + dl, pos[1] - dl:pos[1] + dl]
        similarity = 1.0 - np.sqrt(np.mean(np.power(cv2.absdiff(img_part, self.tl_template), 2)))
        return similarity

    def calc_tl_img_template(self):
        tl_template = np.mean(self.tl_imgs, 0)
        tl_template = tl_template * 255
        tl_template = tl_template.astype(np.uint8)
        cv2.imwrite("taillight_template.png", tl_template)

    def associate_taillights(self, taillight_positions, diff_img):
        if len(taillight_positions) < 2:
            return []
        elif len(taillight_positions) >= 2:
            p1y = taillight_positions[0][0]
            p1x = taillight_positions[0][1]
            for p2y, p2x in taillight_positions[1:]:
                if np.abs(p1y - p2y) < 50 and np.abs(p1x - p2x) < 200:
                    break
            else:
                logging.debug(f"No match found for ({p1x}/{p1y})!")
                return []
            w = np.abs(p1x - p2x) * 1.4
            h = w * 0.8
            x = (p1x + p2x) / 2
            y = (p1y + p2y) / 2 + h * 0.12
            conf = (diff_img[p1y, p1x] + diff_img[p2y, p2x]) / 2
            tll = Box.from_xywh([p1x, p1y, 10, 10])
            tlr = Box.from_xywh([p2x, p2y, 10, 10])
            if p1x > p2x:
                tll, tlr = tlr, tll

            return [Detection(Box.from_xywh([x, y, w, h]), conf, tll, tlr)]
        else:
            return []

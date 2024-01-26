import numpy as np
import cv2
from detection.CarDetector import CarDetector
from detection.Detection import Detection
from util.Box import Box


class MobileNetSSDCarDetector(CarDetector):
    """Car detector using MobileNetSSD model"""

    def __init__(self,
                 prototxt="detection/model/MobileNetSSD/MobileNetSSD_deploy.prototxt",
                 caffemodel="detection/model/MobileNetSSD/MobileNetSSD_deploy.caffemodel"):
        self.car_detection_network = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.car_detection_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_cars(self, img: np.ndarray) -> [Detection]:
        return (self.detect_cars_batch([img]) or [[]])[0]

    def detect_cars_batch(self, imgs: [np.ndarray]) -> [[Detection]]:
        if len(imgs) == 0:
            return [[]]

        blobs = np.empty((len(imgs), 3, 300, 300), dtype=int)
        car_list = [[] for _ in range(len(imgs))]

        for idx, image in enumerate(imgs):
            blob = cv2.dnn.blobFromImage(image, size=(300, 300), ddepth=cv2.CV_8U)
            blobs[idx, :, :, :] = blob

        self.car_detection_network.setInput(blobs, scalefactor=1.0 / 127.5, mean=[127.5, 127.5, 127.5])
        detections = self.car_detection_network.forward()

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence > 0.4:
                # extract the index of the class label from the detections list
                class_idx = int(detections[0, 0, i, 1])

                # if the class label is not "car" (index 7), ignore it
                if class_idx != 7:
                    continue

                image_idx = int(detections[0, 0, i, 0])
                height, width, channels = imgs[image_idx].shape

                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                box = box.astype("int")
                car_list[image_idx].append(Detection(Box.from_ltrb(box), confidence))

        return car_list

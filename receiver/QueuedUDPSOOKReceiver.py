import cv2
import numpy as np

from classification.TaillightClassifier import TaillightClassifier
from detection.Detection import Detection
from detection.TaillightDetector import TaillightDetector
from receiver.UDPSOOKReceiver import UDPSOOKReceiver
from tracking.MultiCarTracker import MultiCarTracker
from tracking.TrackedCar import TrackedCar


class QueuedUDPSOOKReceiver(UDPSOOKReceiver):

    def __init__(self,
                 tracker: MultiCarTracker,
                 tl_detector: TaillightDetector,
                 tl_classifier: TaillightClassifier):
        super().__init__(tracker, tl_detector, tl_classifier)
        self.queue_len = 0
        self.max_queue_len = 99
        self.tl_imgs: {int, ([np.ndarray], [np.ndarray])} = {}
        self.last_tl_imgs: {int, ([np.ndarray], [np.ndarray])} = {}

    def update(self) -> None:
        self.cache_general_data()
        img = self.tracker.get_current_img()
        last_img = self.tracker.get_last_img()
        cars = self.tracker.get_tracked_cars()
        car = next((car for car in cars if car.relevant), None)
        if car:
            if not car.receiving:
                car.receiving = True
                car.start_frame = self.frame_id - self.tracker.get_tracking_delay()

            self.cache_general_car_data(car)

            tll, tlr = self.tl_detector.detect_taillights(img, car.detection)

            if car.id in self.tl_imgs.keys():
                imgs_tll, imgs_tlr = self.tl_imgs[car.id]
                imgs_tll.append(tll.box.crop_from(img))
                imgs_tlr.append(tlr.box.crop_from(img))

                last_imgs_tll, last_imgs_tlr = self.last_tl_imgs[car.id]
                last_imgs_tll.append(tll.box.crop_from(last_img))
                last_imgs_tlr.append(tlr.box.crop_from(last_img))
            else:
                self.tl_imgs[car.id] = ([tll.box.crop_from(img)], [tlr.box.crop_from(img)])
                self.last_tl_imgs[car.id] = ([tll.box.crop_from(last_img)], [tlr.box.crop_from(last_img)])
            self.queue_len += 1

        if self.queue_len >= self.max_queue_len:
            self.run_receiver()

    def run_receiver(self):
        all_tl_imgs = []
        for key in self.tl_imgs.keys():
            imgs_tll, imgs_tlr = self.tl_imgs[key]
            all_tl_imgs.extend(imgs_tll)
            all_tl_imgs.extend(imgs_tlr)

        all_tl_states = self.tl_classifier.classify_taillight_batch(all_tl_imgs)
        for key in self.tl_imgs.keys():
            imgs_tll, imgs_tlr = self.tl_imgs[key]
            states_tll_new = all_tl_states[:len(imgs_tll)]
            all_tl_states = all_tl_states[len(imgs_tll):]
            states_tlr_new = all_tl_states[:len(imgs_tlr)]
            all_tl_states = all_tl_states[len(imgs_tlr):]

            if key in self.states.keys():
                states_tll, states_tlr = self.states[key]
                states_tll.extend(states_tll_new)
                states_tlr.extend(states_tlr_new)
            else:
                self.states[key] = (states_tll_new, states_tlr_new)

        assert (len(all_tl_states) == 0)
        self.tl_imgs = {}
        self.queue_len = 0

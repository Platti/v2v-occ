import cv2
import numpy as np

from classification.TaillightClassifier import TaillightClassifier
from classification.training.ConsecutiveImagesObject import ConsecutiveImagesObject
from util.Util import pairwise


class CNNTaillightStateChangeClassifier(TaillightClassifier):
    def __init__(self, model_file="classification/model/taillight-state-change-classification.h5"):
        from keras.models import load_model as load_keras_model
        self.model = load_keras_model(model_file)
        self.model.summary()

    def classify_taillight(self, img: np.ndarray, last_img: np.ndarray = None, tl_id: int = 0) -> bool:
        if img is None or 0 in img.shape or last_img is None:
            return False

        cnn_input = ConsecutiveImagesObject.create_cnn_input(last_img, img)
        result = self.model.predict(np.array([cnn_input]), verbose=0)
        return np.argmax(result) == 1

    def classify_taillight_batch(self, imgs: [np.array], last_imgs: [np.array] = None) -> [bool]:
        if last_imgs is None or len(imgs) != len(last_imgs) or len(imgs) < 2:
            return []

        cnn_inputs = []
        for last_img, img in zip(last_imgs, imgs):
            cnn_inputs.append(ConsecutiveImagesObject.create_cnn_input(last_img, img))
        result = self.model.predict(np.array(cnn_inputs), verbose=0)
        return (np.argmax(result, axis=1) == 1).tolist()

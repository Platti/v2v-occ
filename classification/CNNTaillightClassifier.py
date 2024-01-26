import cv2
import numpy as np

from classification.TaillightClassifier import TaillightClassifier


class CNNTaillightClassifier(TaillightClassifier):
    def __init__(self, model_file="classification/model/taillight-state-classification.h5"):
        from keras.models import load_model as load_keras_model
        self.model = load_keras_model(model_file)
        self.model.summary()

    def classify_taillight(self, img: np.ndarray, last_img: np.ndarray = None, tl_id: int = 0) -> bool:
        img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        result = self.model.predict(np.array([img]))
        return np.argmax(result) == 1

    def classify_taillight_batch(self, imgs: [np.array], last_imgs: [np.array] = None) -> [bool]:
        if len(imgs) == 0:
            return []

        resized_imgs = []
        for img in imgs:
            height, width, channels = img.shape
            if height == 0 or width == 0:
                resized_imgs.append(np.zeros((28, 28, 3)))
            else:
                resized_imgs.append(cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC))
        result = self.model.predict(np.array(resized_imgs))
        return (np.argmax(result, axis=1) == 1).tolist()

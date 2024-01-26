from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class ConsecutiveImagesObject:
    img1: np.ndarray
    img2: np.ndarray
    bit: bool

    @staticmethod
    def create_cnn_input(img1, img2) -> np.ndarray:
        try:
            img1 = cv2.resize(img1, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            diff_img = cv2.absdiff(img1, img2)
            return np.concatenate((img1, img2, diff_img), axis=2)
        except Exception as e:
            return np.zeros((28, 28, 9))

    @property
    def cnn_input(self) -> np.ndarray:
        cnn_input = ConsecutiveImagesObject.create_cnn_input(self.img1, self.img2)
        assert cnn_input.shape == (28, 28, 9)
        return cnn_input

    @property
    def one_hot_label(self) -> []:
        return [0, 1] if self.bit else [1, 0]

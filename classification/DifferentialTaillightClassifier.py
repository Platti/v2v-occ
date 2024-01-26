import cv2
import numpy as np

from classification.TaillightClassifier import TaillightClassifier


class DifferentialTaillightClassifier(TaillightClassifier):
    def __init__(self):
        self.last_tl_diffs = {}

    def classify_taillight(self, img: np.ndarray, last_img: np.ndarray = None, tl_id: int = 0) -> bool:
        if img is None or 0 in img.shape or last_img is None:
            return False

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_img = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)
        # last_img = cv2.resize(last_img, img.shape[1::-1])

        diff = cv2.absdiff(last_img, img)
        max_diff = np.max(diff)
        mean_max_diff = np.mean(np.sort(diff, axis=None)[-30:])

        last_mean_max_diff, last_ret = self.last_tl_diffs[tl_id] if tl_id in self.last_tl_diffs else (0, False)

        if mean_max_diff > 90:
            ret = True
        elif mean_max_diff < 30:
            ret = False
        elif mean_max_diff - last_mean_max_diff > 20:
            ret = True
        elif mean_max_diff - last_mean_max_diff < -20:
            ret = False
        else:
            ret = last_ret

        if True or 60 < max_diff < 90:
            print(f"Maximum Difference: {max_diff:3.0f}  Mean Maximum Difference: {mean_max_diff:3.1f}  Returns: {ret}")
            cv2.imshow("tl_diff", cv2.resize(diff, None, fx=5, fy=5))
            cv2.imshow("last_img", cv2.resize(last_img, None, fx=5, fy=5))
            cv2.imshow("img", cv2.resize(img, None, fx=5, fy=5))
            #cv2.waitKey(0)

        self.last_tl_diffs[tl_id] = mean_max_diff, ret
        return ret

    def classify_taillight_batch(self, imgs: [np.array], last_imgs: [np.array] = None) -> [bool]:
        return []

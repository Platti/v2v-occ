import numpy as np

from classification.TaillightClassifier import TaillightClassifier
from detection.TaillightDetector import TaillightDetector
from receiver.QueuedUDPSOOKReceiver import QueuedUDPSOOKReceiver
import evaluation.BERAnalyzer as ber_analyzer
from tracking.MultiCarTracker import MultiCarTracker


class QueuedDifferentialUDPSOOKReceiver(QueuedUDPSOOKReceiver):
    def run_receiver(self):
        for key in self.tl_imgs.keys():
            imgs_tll, imgs_tlr = self.tl_imgs[key]
            last_imgs_tll, last_imgs_tlr = self.last_tl_imgs[key]
            states_tll_new = self.tl_classifier.classify_taillight_batch(imgs_tll, last_imgs=last_imgs_tll)
            states_tlr_new = self.tl_classifier.classify_taillight_batch(imgs_tlr, last_imgs=last_imgs_tlr)
            self.tl_imgs[key] = ([], [])
            self.last_tl_imgs[key] = ([], [])

            if key in self.states.keys():
                states_tll, states_tlr = self.states[key]
                states_tll.extend(states_tll_new)
                states_tlr.extend(states_tlr_new)
            else:
                self.states[key] = (states_tll_new, states_tlr_new)

        self.queue_len = 0

    def interpret_states(self):
        for key in self.states.keys():
            data_tll, data_tlr = self.states[key]
            self.data_str[key] = "".join(["1" if j else "0" for i in zip(data_tll, data_tlr) for j in i])
            if len(self.data_str[key]) > 400:
                bit_matches, channel_offset, moving_ber, overall_ber = ber_analyzer.eval_ber_for(
                    ber_analyzer.bit_str2arr(self.data_str[key]))
                print(f"Car: {key}\nData: {self.data_str[key]}\nBER: {overall_ber * 100:.3f}%")

    def get_bit_from_states(self, last_state, cur_state):
        return 1 if cur_state else 0

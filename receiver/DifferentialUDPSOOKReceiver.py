import numpy as np

from receiver.UDPSOOKReceiver import UDPSOOKReceiver
import evaluation.BERAnalyzer as ber_analyzer


class DifferentialUDPSOOKReceiver(UDPSOOKReceiver):
    def interpret_states(self):
        for key in self.states.keys():
            data_tll, data_tlr = self.states[key]
            self.data_str[key] = "".join(["1" if j else "0" for i in zip(data_tll, data_tlr) for j in i])
            if len(self.data_str[key]) > 400:
                bit_matches, channel_offset, moving_ber, overall_ber = ber_analyzer.eval_ber_for(
                    ber_analyzer.bit_str2arr(self.data_str[key]))
                print(f"Car: {key}\nData: {self.data_str[key]}\nBER: {overall_ber*100:.3f}%")

    def get_bit_from_states(self, last_state, cur_state):
        return 1 if cur_state else 0

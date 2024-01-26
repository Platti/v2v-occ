import numpy as np
import pandas as pd

import evaluation.BERAnalyzer as ber_analyzer
from evaluation.plot.OverviewPlotter import create_figure


class OverviewCSVEvaluator:
    csv_path: str

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path, true_values=['True'], false_values=['False'])

    def eval_bit_errors(self):
        for car_id in self.data[(self.data.bit_tll != -1)].car_id.unique():
            mask = (self.data.car_id == car_id) & (self.data.bit_tll != -1) & (self.data.bit_tlr != -1)
            car_data = self.data[mask]
            bit_list = []
            for bit_tll, bit_tlr in zip(car_data['bit_tll'], car_data['bit_tlr']):
                bit_list.append(bit_tll)
                bit_list.append(bit_tlr)
            if len(bit_list) > 400:
                bit_matches, channel_offset, moving_ber, overall_ber = ber_analyzer.eval_ber_for(np.array(bit_list))
                self.data.loc[mask, 'correct_tll'] = bit_matches[::2]
                self.data.loc[mask, 'correct_tlr'] = bit_matches[1::2]
                self.data.loc[mask, 'offset_tll'] = channel_offset[::2]
                self.data.loc[mask, 'offset_tlr'] = channel_offset[1::2]
                self.data.loc[mask, 'ber'] = moving_ber[::2]
                print(f'Car {car_id}: BER = {overall_ber:.03f}')
                print(f'Car {car_id}: BER = {100 * overall_ber:.06f}%')
                print(f'Car {car_id}: BER Left = {100 * (1 - bit_matches[::2].sum() / len(bit_matches[::2])):.06f}%')
                print(f'Car {car_id}: BER Right = {100 * (1 - bit_matches[1::2].sum() / len(bit_matches[1::2])):.06f}%')

        with open(self.csv_path, "w", newline="") as file:
            self.data.to_csv(file, index=False)

        fig = create_figure(self.data)

        return fig, self.data


if __name__ == '__main__':
    test = OverviewCSVEvaluator(
        "C:/Users/p27661/Documents/FH offline/Dissertation/taillight-detection/output/data/DJI_0286_001.MP4_TS1650973191755/overview.csv")
    fig, _ = test.eval_bit_errors()
    fig.show()

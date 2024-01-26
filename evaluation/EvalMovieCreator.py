import cv2
import numpy as np
import pandas as pd

import matplotlib as mpl
from itertools import cycle
from detection.Detection import Detection
from detection.StaticTaillightDetector import StaticTaillightDetector
from evaluation.OverviewCSVEvaluator import OverviewCSVEvaluator
from util import Util
from util.Box import Box
from visualization.Visualizer import Visualizer

from evaluation.plot.OverviewPlotter import create_figure


class EvalMovieCreator:
    def __init__(self, csv_path, video_path, output_path=None):
        self.csv_path = csv_path
        self.video_path = video_path
        self.output_path = output_path

    """
    target_fps: DJI=30, Tesla=36
    tl_state_format_state: data stored actual state=True, data stored state change=False
    """

    def create(self, target_fps=30, tl_state_format_state=True, start_frame_id=1):
        csv_evaluator = OverviewCSVEvaluator(self.csv_path)
        figure, data = csv_evaluator.eval_bit_errors()
        figure = create_figure(data,
                               plot_brightness=True,
                               plot_distance=True,
                               plot_speed=False,
                               plot_offset=False,
                               plot_error=True,
                               plot_ber=True,
                               plot_transmission_time_rs=[],
                               plot_legend=False,
                               fps=target_fps)

        # figure.axes[-1].set_xlim([
        #    pd.to_datetime(-0.05, unit='m'),
        #    pd.to_datetime(5, unit='m')
        # ])

        colors = cycle(mpl.rcParams['axes.prop_cycle'])
        next(colors)
        next(colors)
        color = (255, 255, 255)

        tl_detector = StaticTaillightDetector()

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id-1)

        video_writer = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        temp_vlines = []
        frame_id = start_frame_id
        fig_img = None
        car_id = -1
        car_ids = data[(data.car_id != -1) & ~np.isnan(data.ber)].car_id.unique()

        last_tll_state = False
        last_tlr_state = False

        while frame_id < len(data):
            ret, frame = cap.read()
            frame = Util.crop_frame(frame)

            data_row = data.iloc[frame_id]

            if car_id != data_row.car_id and (data_row.car_id in car_ids):
                color_hex = next(colors)['color'][1:]
                color = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))[::-1]
                car_id = data_row.car_id

            if data_row.car_id in car_ids:
                car_detection = Detection(
                    box=Box.from_ltwh(
                        (data_row.left, data_row.top, data_row.width, data_row.height)),
                    confidence=data_row.confidence)
                frame = Visualizer.draw_car_on(img=frame, car_id=data_row.car_id,
                                               detection=car_detection,
                                               color=color, color_text=(0, 0, 0))

                tll, tlr = tl_detector.detect_taillights(frame, car_detection)
                if tl_state_format_state:
                    frame = Visualizer.draw_taillights_on(img=frame, tll_detection=tll, tlr_detection=tlr,
                                                          tll_state=data_row.state_tll is True or data_row.state_tll == "True",
                                                          tlr_state=data_row.state_tlr is True or data_row.state_tlr == "True")
                else:
                    frame = Visualizer.draw_taillights_transition_on(img=frame, tll_detection=tll, tlr_detection=tlr,
                                                                     tll_state=data_row.state_tll != str(
                                                                         last_tll_state),
                                                                     tlr_state=data_row.state_tlr != str(
                                                                         last_tlr_state))
                    last_tll_state = data_row.state_tll != str(last_tll_state)
                    last_tlr_state = data_row.state_tlr != str(last_tlr_state)
            frame = Visualizer.draw_frame_id(frame, frame_id)

            if fig_img is None or frame_id % 10 == 0:
                time = pd.to_datetime(frame_id / target_fps, unit='s')
                print(time)
                for ax in figure.axes:
                    temp_vlines.append(ax.axvline(time, color="black", zorder=-1))
                figure.canvas.draw()
                for vline in temp_vlines:
                    vline.remove()
                temp_vlines.clear()

                fig_img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
                fig_img = fig_img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
                fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR)

            total_img = np.zeros((480, 1470, 3), dtype=np.uint8)
            total_img[15:465, 15:815] = frame
            total_img[0:480, 830:1470] = fig_img
            cv2.imshow("plot", total_img)
            cv2.waitKey(1)

            if self.output_path:
                if not video_writer:
                    height, width, channels = total_img.shape
                    video_writer = cv2.VideoWriter(self.output_path, fourcc, target_fps, (width, height))

                video_writer.write(total_img)

            frame_id += 1

import os
import time
from enum import Enum

from util.Box import Box
import numpy as np
import itertools
import pandas as pd


def current_time_ms():
    return round(time.time() * 1000)


def crop_frame(frame) -> np.array:
    height, width, channels = frame.shape
    # if frame is 4K
    if height == 2160 and width == 3840:
        # box = Box.from_ltwh([1550, 950, 800, 450])
        # box = Box.from_ltwh([1550, 1250, 800, 450])
        box = Box.from_ltwh([1300, 1150, 800, 450])
        return box.crop_from(frame)
    else:
        return frame


def pairwise(iterable):
    """
    pairwise iterator: s = (s0,s1), (s1,s2), (s2,s3),...
    added to itertools in Python 3.10
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class ConfigTracking(Enum):
    MNSSD = 1
    BRIGHTNESS = 2
    DUAL = 3
    MANUAL = 4


class ConfigReceiving(Enum):
    CNNSTATE = 1
    CNNSTATECHANGE = 2
    BRIGHTNESSCHANGE = 3


def collect_data(path: str, videos: [str], conf_track: ConfigTracking, conf_rec: ConfigReceiving,
                 filenames: [str] = ["data.csv"]):
    data = pd.DataFrame()
    dirs = os.listdir(path)

    for video in videos:
        video_dirs = [vd for vd in dirs if video in vd and str(conf_track) in vd and vd.endswith(str(conf_rec))]
        if len(video_dirs) != 1:
            print(f"CAUTION! Expected 1 result directory but found {len(video_dirs)} for {video}.")
            return None
        print(f"Loading data for video {video}...")
        data_video = pd.read_csv(f"{path}{video_dirs[0]}/{filenames[0]}")
        if len(filenames) == 2:
            data_video_2 = pd.read_csv(f"{path}{video_dirs[0]}/{filenames[1]}")
            data_video = pd.merge(data_video, data_video_2, on=[column for column in data_video.columns.values if
                                                                "transmission_time" not in column])
        data = pd.concat([data, data_video], axis=0)

    return data


ALL_DJI_VIDEO_NAMES = ["DJI_0282.mp4",
                       "DJI_0283.mp4",
                       "DJI_0284.mp4",
                       "DJI_0285.mp4",
                       "DJI_0286.mp4",
                       "DJI_0287.mp4",
                       "DJI_0288.mp4",
                       "DJI_0289.mp4",
                       "DJI_0290.mp4",
                       "DJI_0291.mp4",
                       "DJI_0292.mp4",
                       "DJI_0293.mp4",
                       "DJI_0294.mp4",
                       "DJI_0437.mp4",
                       "DJI_0438.mp4",
                       "DJI_0439.mp4",
                       "DJI_0441.mp4",
                       ]

ALL_TESLA_VIDEO_NAMES = ["tesla_230201.mp4",
                         "tesla_230207.mp4",
                         "tesla_230208.mp4"
                         ]

ALL_EXP_DISTANCE_NAMES = ["DJI_0228.MP4",
                          "DJI_0231.MP4",
                          "DJI_0233.MP4",
                          "DJI_0236.MP4",
                          "DJI_0238.MP4",
                          "DJI_0240.MP4",
                          "DJI_0243.MP4",
                          "DJI_0245.MP4"
                          ]

ALL_EXP_PWR_NAMES = ["DJI_0249.MP4",
                     "DJI_0250.MP4",
                     "DJI_0251.MP4",
                     "DJI_0253.MP4",
                     "DJI_0254.MP4",
                     "DJI_0257.MP4",
                     "DJI_0258.MP4",
                     "DJI_0259.MP4",
                     "DJI_0260.MP4",
                     "DJI_0261.MP4",
                     "DJI_0263.MP4",
                     "DJI_0264.MP4",
                     "DJI_0265.MP4",
                     "DJI_0266.MP4",
                     "DJI_0267.MP4",
                     "DJI_0268.MP4",
                     "DJI_0269.MP4",
                     "DJI_0270.MP4",
                     "DJI_0271.MP4",
                     "DJI_0273.MP4",
                     "DJI_0274.MP4",
                     "DJI_0275.MP4"
                     ]

FIG_WIDTH = 6
FIG_HEIGHT = 3

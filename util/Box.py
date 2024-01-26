from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @staticmethod
    def from_ltrb(ltrb):
        return Box(
            int(ltrb[0]),
            int(ltrb[1]),
            int(ltrb[2]),
            int(ltrb[3]))

    @staticmethod
    def from_ltwh(ltwh):
        l, t, w, h = int(ltwh[0]), int(ltwh[1]), int(ltwh[2]), int(ltwh[3])
        r = l + w
        b = t + h
        return Box(l, t, r, b)

    @staticmethod
    def from_xywh(xywh):
        x, y, w, h = int(xywh[0]), int(xywh[1]), int(xywh[2]), int(xywh[3])
        l = x - w // 2
        t = y - h // 2
        r = l + w
        b = t + h
        return Box(l, t, r, b)

    @property
    def area(self) -> int:
        return max(0, self.width * self.height)

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def aspect_ratio(self) -> float:
        return float("inf") if self.height == 0 else self.width / self.height

    def align(self) -> None:
        """align left/right for testing perspective at angle"""
        #self.left = self.right - self.height * 11//10
        self.right = self.left + self.height * 11//10

    def calc_intersection(self, other: Box) -> Box:
        return Box(
            max(self.left, other.left),
            max(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom))

    def calc_iou(self, other: Box) -> float:
        intersection = self.calc_intersection(other)
        return intersection.area / (self.area + other.area - intersection.area)

    def calc_intersection_over_area(self, other: Box) -> float:
        intersection = self.calc_intersection(other)
        return intersection.area / self.area

    def calc_overlapping_rate(self, other: Box) -> float:
        intersection = self.calc_intersection(other)
        min_box_area = min(self.area, other.area)
        return intersection.area / min_box_area

    def as_ltwh(self) -> [int]:
        return [self.left, self.top, self.width, self.height]

    def as_ltrb(self) -> [int]:
        return [self.left, self.top, self.right, self.bottom]

    def combine(self, other: Box) -> None:
        self.left = min(other.left, (self.left + other.left) // 2)
        self.top = min(other.top, (self.top + other.top) // 2)
        self.right = max(other.right, (self.right + other.right) // 2)
        self.bottom = max(other.bottom, (self.bottom + other.bottom) // 2)

    def reset_origin(self, x: int, y: int) -> None:
        self.left = self.left + x
        self.top = self.top + y
        self.right = self.right + x
        self.bottom = self.bottom + y

    def crop_from(self, image: np.ndarray) -> np.ndarray:
        return image[self.top:self.bottom, self.left:self.right]

    def __str__(self):
        return f"Box(l:{self.left}, t:{self.top}, r:{self.right}, b:{self.bottom})"

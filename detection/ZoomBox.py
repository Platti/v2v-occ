from util.Box import Box


class ZoomBox:
    """Wrapper for Box with weight"""

    def __init__(self, box: Box, weight: float):
        self.box = box
        self.weight = weight

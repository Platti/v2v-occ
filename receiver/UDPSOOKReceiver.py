import cv2

from classification.TaillightClassifier import TaillightClassifier
from detection.Detection import Detection
from detection.TaillightDetector import TaillightDetector
from tracking.MultiCarTracker import MultiCarTracker
from util.Observer import TrackerObserver
import evaluation.BERAnalyzer as ber_analyzer

import pandas as pd


class UDPSOOKReceiver(TrackerObserver):
    def __init__(self,
                 tracker: MultiCarTracker,
                 tl_detector: TaillightDetector,
                 tl_classifier: TaillightClassifier):
        super().__init__(tracker)
        self.frame_id = 0
        self.tl_detector = tl_detector
        self.tl_classifier = tl_classifier
        self.states: {int, ([bool], [bool])} = {}
        self.data_str: {int, str} = {}
        self.detections: {int, ([int], [int], [int], [int], [float], [int])} = {}
        self.brightness: [int] = []

    def update(self) -> None:
        self.cache_general_data()
        last_img = self.tracker.get_last_img()
        img = self.tracker.get_current_img()
        cars = self.tracker.get_tracked_cars()
        car = next((car for car in cars if car.relevant), None)
        if car:
            if not car.receiving:
                car.receiving = True
                car.start_frame = self.frame_id - self.tracker.get_tracking_delay()

            self.cache_general_car_data(car)

            tll, tlr = self.tl_detector.detect_taillights(img, car.detection)
            #state_tll, state_tlr = self.tl_classifier.classify_taillight_batch([tll.box.crop_from(img),
            #                                                                    tlr.box.crop_from(img)])
            state_tll = self.tl_classifier.classify_taillight(tll.box.crop_from(img), tll.box.crop_from(last_img), 1)
            state_tlr = self.tl_classifier.classify_taillight(tlr.box.crop_from(img), tlr.box.crop_from(last_img), 2)
            #state_tlr = False
            if car.id in self.states.keys():
                states_tll, states_tlr = self.states[car.id]
                states_tll.append(state_tll)
                states_tlr.append(state_tlr)
            else:
                self.states[car.id] = ([state_tll], [state_tlr])

    def cache_general_data(self):
        self.frame_id += 1
        img = self.tracker.get_current_img()
        self.brightness.append(UDPSOOKReceiver.calc_brightness(img))

    def cache_general_car_data(self, car):
        img = self.tracker.get_current_img()
        car_img = car.detection.box.crop_from(img)
        if car.id in self.detections.keys():
            l, t, w, h, c, b = self.detections[car.id]
            l.append(car.detection.box.left)
            t.append(car.detection.box.top)
            w.append(car.detection.box.width)
            h.append(car.detection.box.height)
            c.append(car.detection.confidence)
            b.append(UDPSOOKReceiver.calc_brightness(car_img))
        else:
            self.detections[car.id] = ([car.detection.box.left],
                                       [car.detection.box.top],
                                       [car.detection.box.width],
                                       [car.detection.box.height],
                                       [car.detection.confidence],
                                       [UDPSOOKReceiver.calc_brightness(car_img)])

    @staticmethod
    def calc_brightness(img):
        if img is None or 0 in img.shape:
            return -1
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            return int(v.mean())

    def interpret_states(self):
        for key in self.states.keys():
            states_tll, states_tlr = self.states[key]
            data_tll = UDPSOOKReceiver.decode(states_tll)
            data_tlr = UDPSOOKReceiver.decode(states_tlr)
            self.data_str[key] = "".join(["1" if j else "0" for i in zip(data_tll, data_tlr) for j in i])

            if len(self.data_str[key]) > 400:
                bit_matches, channel_offset, moving_ber, overall_ber = ber_analyzer.eval_ber_for(
                    ber_analyzer.bit_str2arr(self.data_str[key]))
                print(f"Car: {key}\nData: {self.data_str[key]}\nBER: {overall_ber * 100:.3f}%")

    @staticmethod
    def decode(states: [bool]):
        return [states[i] != states[i + 1] for i in range(len(states) - 1)]

    def store_data_files(self, output_dir: str):
        self.interpret_states()
        for key in self.data_str.keys():
            with open(f"{output_dir}/data_{key:03.0f}.txt", "w") as file:
                file.write(self.data_str[key])

    def store_data_overview(self, total_frames: int, output_dir: str):
        overview = pd.DataFrame(
            {'frame_id': pd.Series(data=range(total_frames), dtype=pd.Int32Dtype),
             'car_id': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'state_tll': pd.Series(data=[-1] * total_frames, dtype=pd.Int8Dtype),
             'state_tlr': pd.Series(data=[-1] * total_frames, dtype=pd.Int8Dtype),
             'bit_tll': pd.Series(data=[-1] * total_frames, dtype=pd.Int8Dtype),
             'bit_tlr': pd.Series(data=[-1] * total_frames, dtype=pd.Int8Dtype),
             'correct_tll': pd.Series(data=[-1] * total_frames, dtype=pd.Int8Dtype),
             'correct_tlr': pd.Series(data=[-1] * total_frames, dtype=pd.Int8Dtype),
             'offset_tll': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'offset_tlr': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'left': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'top': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'width': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'height': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'confidence': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'brightness': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'brightness_car': pd.Series(data=[-1] * total_frames, dtype=pd.Int32Dtype),
             'ber': pd.Series(data=[-1.0] * total_frames, dtype=pd.Float32Dtype)})
        frame_id = 0
        car = None
        car_list = self.tracker.get_tracked_cars_history()

        while frame_id < total_frames:
            while car is None or not car.receiving:
                if car_list:
                    car = car_list.pop(0)
                else:
                    car = None
                    break
            if car is None:
                break
            next_start_frame_id = car.start_frame
            while frame_id < next_start_frame_id:
                overview.loc[frame_id] = overview_entry(frame_id=frame_id,
                                                        brightness=self.get_brightness_in_frame(frame_id))
                frame_id += 1
            car_id = car.id

            if car_id in self.states.keys():
                states_tll, states_tlr = self.states[car_id]
                l, t, w, h, c, b = self.detections[car_id]
                last_state_tll = False
                last_state_tlr = False
                for state_tll, state_tlr, l, t, w, h, c, b in zip(states_tll, states_tlr, l, t, w, h, c, b):
                    overview.loc[frame_id] = overview_entry(frame_id=frame_id,
                                                            brightness=self.get_brightness_in_frame(frame_id),
                                                            brightness_car=b,
                                                            car_id=car_id,
                                                            state_tll=state_tll, state_tlr=state_tlr,
                                                            bit_tll=self.get_bit_from_states(last_state_tll, state_tll),
                                                            bit_tlr=self.get_bit_from_states(last_state_tlr, state_tlr),
                                                            left=l, top=t, width=w, height=h, confidence=c)
                    last_state_tll = state_tll
                    last_state_tlr = state_tlr
                    frame_id += 1
            car = None

        while frame_id < total_frames:
            overview.loc[frame_id] = overview_entry(frame_id=frame_id,
                                                    brightness=self.get_brightness_in_frame(frame_id))
            frame_id += 1

        with open(f"{output_dir}/overview.csv", "w", newline="") as file:
            overview.to_csv(file, index=False)

    def get_bit_from_states(self, last_state, cur_state):
        return 0 if last_state == cur_state else 1

    def get_brightness_in_frame(self, frame_id):
        if frame_id < len(self.brightness):
            return self.brightness[frame_id]
        else:
            return self.brightness[-1]


def overview_entry(frame_id, car_id=-1, state_tll=-1, state_tlr=-1, bit_tll=-1, bit_tlr=-1, correct_tll=-1,
                   correct_tlr=-1, offset_tll=-1, offset_tlr=-1, left=-1, top=-1, width=-1, height=-1, confidence=0,
                   brightness=-1, brightness_car=-1, ber='NaN'):
    return {'frame_id': frame_id,
            'car_id': car_id,
            'state_tll': state_tll,
            'state_tlr': state_tlr,
            'bit_tll': bit_tll,
            'bit_tlr': bit_tlr,
            'correct_tll': correct_tll,
            'correct_tlr': correct_tlr,
            'offset_tll': offset_tll,
            'offset_tlr': offset_tlr,
            'left': left,
            'top': top,
            'width': width,
            'height': height,
            'confidence': confidence,
            'brightness': brightness,
            'brightness_car': brightness_car,
            'ber': ber}

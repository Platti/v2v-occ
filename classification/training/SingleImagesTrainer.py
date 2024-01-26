import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Lambda

from classification.training.AbstractTrainer import AbstractTrainer
from util.Util import current_time_ms


def create_model_v1():
    model = Sequential(name="taillight-state-recognition")
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(28, 28, 3)))
    model.add(Conv2D(6, (5, 5), activation='relu', padding="same", input_shape=(28, 28, 3), name="image"))
    model.add(Conv2D(12, (5, 5), activation='relu', strides=2, padding="same"))
    model.add(Conv2D(24, (4, 4), activation='relu', strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(200, activation='relu'))
    model.add(Dense(units=2, activation='softmax', name="taillight_state"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


class SingleImagesTrainer(AbstractTrainer):
    def read_dataset(self):
        for data_path in self.data_paths:
            self.read_dataset_single_class(f"{data_path}/OFF/", label=False)
            self.read_dataset_single_class(f"{data_path}/ON/", label=True)

        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

        print(f"Number of inputs and labels: {len(self.inputs)} == {len(self.labels)}")

    def read_dataset_single_class(self, data_path, label):
        for file in os.listdir(data_path):
            if file.endswith(".png"):
                self.inputs.append(cv2.imread(data_path + file, cv2.IMREAD_COLOR))
                self.labels.append([0, 1] if label else [1, 0])

    def get_output_filename(self):
        return f"taillight-state-classification_TS{current_time_ms()}.h5"

    def show_input(self, index):
        pass

    def show_input(self, index):
        img = self.inputs[index].astype(np.uint8)
        label = np.argmax(self.labels, axis=1)[index]
        print(f"Label: {label} =>"
              f" {'ON' if label == 1 else 'OFF'}"
              f" --- Press 'y' to keep, 'x' to invert, 'd' to delete.")
        while True:
            cv2.imshow("input", cv2.resize(img, (280, 280)))
            selection = cv2.waitKey(0)
            if selection == ord('y') or selection == ord('x') or selection == ord('d'):
                break

        if selection == ord('x'):
            print(f"Invert label of data element with index {index}")
        elif selection == ord('d'):
            print(f"Delete data element (move to quarantine folder) with index {index}")


if __name__ == '__main__':
    trainer = SingleImagesTrainer(create_model_v1(),
                                  [
                                      "C:/Users/p27661/OneDrive - FH OOe/Dissertation/samples/final/DJI_0286.MP4_TS1686645357275",
                                      "C:/Users/p27661/OneDrive - FH OOe/Dissertation/samples/final/DJI_0291.MP4_TS1686816880733",
                                      "C:/Users/p27661/OneDrive - FH OOe/Dissertation/samples/final/DJI_0294.MP4_TS1686900038890",
                                      "C:/Users/p27661/OneDrive - FH OOe/Dissertation/samples/final/tesla_230201.mp4_TS1686733097376",
                                      "C:/Users/p27661/OneDrive - FH OOe/Dissertation/samples/final/tesla_230208.mp4_TS1686813997144"
                                  ],
                                  "../model")

    trainer.read_dataset()
    if True:
        trainer.train()
    else:
        trainer.refine_dataset("asdf")

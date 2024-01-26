import os
import pickle

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Lambda

from classification.training.AbstractTrainer import AbstractTrainer
from util.Util import current_time_ms


def create_model_v1():
    model = Sequential(name="taillight-state-change-recognition")
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(28, 28, 6)))
    model.add(Conv2D(6, (5, 5), activation='relu', padding="same", input_shape=(28, 28, 6), name="image"))
    model.add(Conv2D(12, (5, 5), activation='relu', strides=2, padding="same"))
    model.add(Conv2D(24, (4, 4), activation='relu', strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(200, activation='relu'))
    model.add(Dense(units=2, activation='softmax', name="taillight-state-change"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


def create_model_v2():
    model = Sequential(name="taillight-state-change-recognition-with-diff-image")
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(28, 28, 9)))
    model.add(Conv2D(6, (5, 5), activation='relu', padding="same", input_shape=(28, 28, 9), name="image"))
    model.add(Conv2D(12, (5, 5), activation='relu', strides=2, padding="same"))
    model.add(Conv2D(24, (4, 4), activation='relu', strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(200, activation='relu'))
    model.add(Dense(units=2, activation='softmax', name="taillight-state-change"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


class ConsecutiveImagesTrainer(AbstractTrainer):
    def read_dataset(self):
        cios = []

        for data_path in self.data_paths:
            for file in os.listdir(data_path):
                if file.endswith(".cio"):
                    with open(f"{data_path}/{file}", 'rb') as input_file:
                        cios.append(pickle.load(input_file))

        self.inputs = np.array([cio.cnn_input for cio in cios])
        self.labels = np.array([cio.one_hot_label for cio in cios])

        print(f"Number of inputs and labels: {len(self.inputs)} == {len(self.labels)}")

    def get_output_filename(self):
        return f"taillight-state-change-classification_TS{current_time_ms()}.h5"

    def show_input(self, index):
        cnn_input = self.inputs[index]
        img1 = cnn_input[:, :, 0:3].astype(np.uint8)
        img2 = cnn_input[:, :, 3:6].astype(np.uint8)
        label = np.argmax(self.labels, axis=1)[index]
        print(f"Label: {label} =>"
              f" {'STATE CHANGED' if label == 1 else 'STATE DID NOT CHANGE'}"
              f" --- Press 'y' to keep, 'x' to invert, 'd' to delete.")
        while True:
            cv2.imshow("input", cv2.resize(img1, (280, 280)))
            selection = cv2.waitKey(300)
            if selection == ord('y') or selection == ord('x') or selection == ord('d'):
                break
            cv2.imshow("input", cv2.resize(img2, (280, 280)))
            selection = cv2.waitKey(300)
            if selection == ord('y') or selection == ord('x') or selection == ord('d'):
                break

        if selection == ord('x'):
            print(f"Invert label of data element with index {index}")
        elif selection == ord('d'):
            print(f"Delete data element (move to quarantine folder) with index {index}")


if __name__ == '__main__':
    trainer = ConsecutiveImagesTrainer(create_model_v2(),
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
        trainer.refine_dataset("taillight-state-change-classification_TS1686905925539.h5")

from abc import ABC, abstractmethod
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split


class AbstractTrainer(ABC):
    def __init__(self, model, data_paths, output_path):
        self.model = model
        self.data_paths = data_paths
        self.output_path = output_path
        self.inputs = []
        self.labels = []

    @abstractmethod
    def read_dataset(self):
        pass

    @abstractmethod
    def get_output_filename(self):
        pass

    def train(self):
        inputs_train, inputs_test, labels_train, labels_test = train_test_split(self.inputs, self.labels,
                                                                                test_size=0.1,
                                                                                random_state=42)
        self.model.fit(inputs_train, labels_train, epochs=50, batch_size=32)
        output_filename = self.get_output_filename()
        self.model.save(filepath=f"{self.output_path}/{output_filename}", overwrite=True)

        (loss, acc) = self.model.evaluate(inputs_train, labels_train, batch_size=128)
        print(f'Training Data: Loss: {loss}, Accuracy: {acc}')
        (loss, acc) = self.model.evaluate(inputs_test, labels_test, batch_size=128)
        print(f'Testing Data:  Loss: {loss}, Accuracy: {acc}')

        reloaded_model = load_model(f"{self.output_path}/{output_filename}")

        inputs_predict = [inputs_test[0], inputs_test[1], inputs_test[2], inputs_test[3], inputs_test[0]]
        inputs_predict = np.array(inputs_predict)

        prediction = reloaded_model.predict(inputs_predict)
        print(prediction)
        print(np.argmax(prediction, axis=1))

    def refine_dataset(self, model_filename):
        model = load_model(f"{self.output_path}/{model_filename}")

        predictions = np.argmax(model.predict(self.inputs), axis=1)
        labels = np.argmax(self.labels, axis=1)

        incorrect_ids = np.flatnonzero(predictions != labels)
        for i in incorrect_ids:
            if labels[i] == 0:
                self.show_input(i)
        for i in incorrect_ids:
            if labels[i] == 1:
                self.show_input(i)

    @abstractmethod
    def show_input(self, index):
        pass

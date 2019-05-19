from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

from thex_data.data_consts import TARGET_LABEL, class_to_subclass
from thex_data.data_clean import convert_class_vectors


class SubNetwork:

    def __init__(self, classes, X, y):
        """
        Neural network with first layer of length input_length, last layer of output_length, where each output node corresponds to a class in classes
        :param classes: List of classes this network runs on; each label in y coresponds to index in this list
        :param y: DataFrame of labels for these classes
        :param X: DataFrame of features for these classes
        """
        self.input_length = len(list(X))
        self.output_length = len(classes)
        self.classes = classes

        print("Initialzing SubNetwork for classes " + str(self.classes))
        print("input length: " + str(self.input_length))
        print("output length: " + str(self.output_length))

        self.network = self.init_network(X, y)

    def get_sample_weights(self, X, y):
        """
        Get weight of each sample (1/# of samples in class)
        """
        labeled_samples = pd.concat([X, y], axis=1)
        classes = labeled_samples[TARGET_LABEL].unique()
        label_counts = {}
        for c in classes:
            label_counts[c] = labeled_samples.loc[
                labeled_samples[TARGET_LABEL] == c].shape[0]
        sample_weights = []
        for df_index, row in labeled_samples.iterrows():
            class_count = label_counts[row[TARGET_LABEL]]
            sample_weights.append(1 / class_count)
        return np.array(sample_weights)

    def init_network(self, X, y):

        model = Sequential()
        model.add(Dense(12, input_dim=self.input_length, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.output_length, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        sample_weights = self.get_sample_weights(X, y)

        # Convert numeric labels to one-hot encoding (which is what Keras expects)
        y = to_categorical(y.values)
        model.fit(X.values, y, batch_size=4, epochs=20,
                  sample_weight=sample_weights)
        return model

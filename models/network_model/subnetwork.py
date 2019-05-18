from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

from thex_data.data_consts import TARGET_LABEL
from thex_data.data_clean import convert_class_vectors


class SubNetwork:

    def __init__(self, level, classes, X, y, class_labels):
        """
        Neural network with first layer of length input_length, last layer of output_length, where each output node corresponds to a class in classes
        :param level: Integer corresponding to this network's class hierarchy level of data
        :param classes: List of class names to filter on for this SubNetwork
        :param y: Class vectors for each sample, in same order as X
        :param X: Features and values for each sample, same order as y
        :param class_labels: Class names in order of the class vectors in y
        """
        self.class_level = level
        self.input_length = len(list(X))
        self.output_length = len(classes)
        self.level_classes = classes

        print("Initialzing SubNetwork at class hierarcy level " +
              str(self.class_level) + "  for classes " + str(self.level_classes))
        print("All class labels: " + str(class_labels))
        print("input length: " + str(self.input_length))
        print("output length: " + str(self.output_length))

        X, y = self.init_data(X, y, class_labels)
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

        # Convert labels to one-hot encoding (which is what Keras expects)
        y = to_categorical(y.values)
        model.fit(X.values, y, batch_size=4, epochs=20,
                  sample_weight=sample_weights)
        return model

    def init_data(self, X, y, class_labels):

        # Assign training data to this network, if it is on the correct level
        keep_indices = []
        labels = []
        for df_index, row in y.iterrows():
            class_vector = row[TARGET_LABEL]
            added = False
            for class_index, class_name in enumerate(class_labels):
                if class_name in self.level_classes and class_vector[class_index] == 1:
                    # if added == True:
                    #     print("Sample was already assigned " + str(self.level_classes[
                    # labels[-1]]) + " and now it also wants " + str(class_name))
                    added = True
                    labels.append(self.level_classes.index(class_name))
                    keep_indices.append(df_index)

        # Filter X and y to have data at this level alone
        y_level = pd.DataFrame(labels, columns=[TARGET_LABEL])
        X_level = X.loc[keep_indices].reset_index(drop=True)

        return X_level, y_level

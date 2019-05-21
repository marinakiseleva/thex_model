from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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

        print("\n\nInitialzing SubNetwork for classes " + str(self.classes))
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
        model.add(Dense(48, input_dim=self.input_length, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.output_length, activation='softmax'))

        sgd = optimizers.SGD(lr=0.0001, decay=0, momentum=0, nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

        sample_weights = self.get_sample_weights(X, y)
        num_samples = X.shape[0]

        # Split data into Training and Validation
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.25, shuffle=True)
        val_sample_weights = self.get_sample_weights(x_valid, y_valid)
        train_sample_weights = self.get_sample_weights(x_train, y_train)

        # Convert numeric labels to one-hot encoding (which is what Keras expects)
        y_train_codes = to_categorical(y_train.values)
        y_val_codes = to_categorical(y_valid.values)

        # Assign class weights
        class_indices = list(range(len(self.classes)))
        class_weights = compute_class_weight(
            class_weight='balanced', classes=class_indices, y=y[TARGET_LABEL].values)
        d_class_weights = dict(enumerate(class_weights))

        # Set batch size based on number of samples
        epochs = 1000
        batch_size = 24
        if num_samples < 100:
            batch_size = 16

        # sample_weight=train_sample_weights,
        es = EarlyStopping(monitor='val_loss', min_delta=0.00001, mode='auto', verbose=1,
                           patience=20)
        model.fit(x_train.values, y_train_codes, batch_size=batch_size, verbose=0, epochs=epochs,
                  validation_data=(x_valid.values, y_val_codes), class_weight=d_class_weights, callbacks=[es])
        metrics = model.evaluate(x_valid.values, y_val_codes)
        print("Model metrics on valiation set: ")
        print(model.metrics_names)
        print(metrics)
        return model

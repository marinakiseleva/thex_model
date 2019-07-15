import pandas as pd
# import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier

from keras.models import Sequential
from keras.layers import Dense


class NetClassifier(BinaryClassifier):
    """
    Extension of abstract class for binary classifier.
    """

    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        self.input_length = len(list(X))
        self.model = self.get_best_model(X, y)
        return self.model

    def predict(self, X):
        """
        Get the probability of the positive class for each row in DataFrame X. Return probabilities as Numpy column.
        :param x: 2D Numpy array as column with probability of class per row
        """
        preds = list(self.classifier.predict(x=X,  batch_size=1)[0])
        print("Full x predictions")
        print(preds)

        return preds

    def get_best_model(self, X, y):
        """
        Use RandomizedSearchCV to compute best hyperparameters for the model, using passed in X and y
        :return: Tree with parameters corresponding to best performance, already fit to data
        """
        # Get weight of each sample by its class frequency
        labeled_samples = pd.concat([X, y], axis=1)
        # sample_weights = self.get_sample_weights(labeled_samples)
        class_weights = self.get_class_weights(labeled_samples)

        model = Sequential()
        model.add(Dense(112, input_dim=self.input_length, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X,
                  y,
                  epochs=150,
                  batch_size=10,
                  verbose=0,
                  class_weight=class_weights)

        return model

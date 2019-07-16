from models.ensemble_models.ensemble_model.ensemble_model import EnsembleModel
from models.ensemble_models.network_model.net import NetClassifier

import numpy as np


class Network(EnsembleModel):
    """
    Model that consists of K neural networks.
    """

    def __init__(self, **data_args):
        self.name = "OVA Neural Network Model"
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.models = {}

    def create_classifier(self, pos_class, X, y):
        """
        Create Decision Tree classifier for pos_class versus all
        """
        return NetClassifier(pos_class, X, y)

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Single row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            model = self.models[class_name].model
            input_length = self.models[class_name].input_length
            if model is not None:
                x_row = np.reshape(x.values, (1, input_length))
                # [[probability of class = 1]]
                class_probability = model.predict(x_row)[0][0]
            else:
                print("No model - probability = 0.")
                class_probability = 0

            probabilities[class_name] = class_probability

        return probabilities

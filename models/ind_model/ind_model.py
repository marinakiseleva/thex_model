"""
Independent Model
Assumes all classes are independent and are not related by any class hierarchy.

"""

import numpy as np
import pandas as pd
from mainmodel.mainmodel import MainModel
from classifiers.optbinary import OptimalBinaryClassifier
from utilities.utilities import *
from thex_data.data_consts import TARGET_LABEL


class IndModel(MainModel):

    def __init__(self, **data_args):
        """
        """
        self.name = "Independent Model"
        super(IndModel, self).__init__(**data_args)

    def get_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def train_model(self, X_train, y_train):
        """
        Train model using training data
        """
        # All classes are a single disjoint set, so create a binary classifier for
        # each one

        self.models = {}
        for class_index, class_name in enumerate(self.class_labels):
            y_relabeled = self.get_class_data(class_name, y_train)

            # TODO! Need to do this in thex_data instead of here.
            positive_count = y_relabeled.loc[y_relabeled[TARGET_LABEL] == 1].shape[0]
            if positive_count < 9:
                print("WARNING: No model for " + class_name)
                continue

            print("\nClass Model: " + class_name)
            self.models[class_name] = OptimalBinaryClassifier(
                class_name, X_train, y_relabeled)

        return self.models

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            probabilities[class_name] = self.models[class_name].get_class_probability(x)
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = 0.001
                print("EnsembleModel get_class_probabilities NULL probability for " + class_name)

            if probabilities[class_name] < 0.0001:
                # Force min prob to 0.001 for future computation
                probabilities[class_name] = 0.001

        # Normalize
        probabilities = self.normalize(probabilities)

        return probabilities

    def normalize(self, probabilities):
        """
        Normalize across probabilities, treating each as independent. So, each is normalized by dividing by the sum of all probabilities.
        :param probabilities: Dict from class names to probabilities, already normalized across disjoint sets
        """
        total = sum(probabilities.values())
        norm_probabilities = {class_name: prob /
                              total for class_name, prob in probabilities.items()}

        return norm_probabilities

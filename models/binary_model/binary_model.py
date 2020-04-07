"""
Binary Model

Computes probability of each class entirely independently using optimized binary classifiers. Almost the same as Independent Model, except the probabilities are not normalized across classes. Strictly concerned with binary probability of each class.

"""

import numpy as np
import pandas as pd

from mainmodel.mainmodel import MainModel
from classifiers.binary.optbinary import OptimalBinaryClassifier
import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS


class BinaryModel(MainModel):

    def __init__(self, **data_args):
        """
        Initialize Independent Model - call init method of parent class
        """
        self.name = "Binary Classifiers"
        super(BinaryModel, self).__init__(**data_args)

    def train_model(self, X_train, y_train):
        """
        Train model using training data. All classes are a single disjoint set, so create a binary classifier for each one
        """

        self.models = {}
        for class_index, class_name in enumerate(self.class_labels):
            y_relabeled = self.relabel_class_data(class_name, y_train)
            print("\nClass Model: " + class_name)
            self.models[class_name] = OptimalBinaryClassifier(
                class_name, X_train, y_relabeled, self.nb, self.dir)

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
            MIN_PROB = 0.0001  # Min probability to avoid overflow
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = MIN_PROB
                print(self.name + " NULL probability for " + class_name)

            if probabilities[class_name] < MIN_PROB:
                probabilities[class_name] = MIN_PROB

        return probabilities

    def is_true_positive(self, is_class, row, class_index, max_class_name, class_name):
        """
        Determines if the prediction is a true positive for the class class_name (need this to overwrite in Binary)
        """
        return is_class and row[class_index] > 0.5

    def get_true_pos_prob(self, row, class_index, max_class_prob):
        """
        Get true positive probability (need this to overwrite in Binary)
        """
        return row[class_index]

    def compute_baselines(self, class_counts, y):
        """
        Overwrite so that class priors are binary (1/2) instead of proportional to # of classes
        Get random classifier baselines for recall, specificity (negative recall), and precision
        """
        pos_baselines = {}
        neg_baselines = {}
        precision_baselines = {}

        total_count = y.shape[0]

        class_priors = {c: 0.5 for c in self.class_labels}

        for class_name in self.class_labels:
            # Compute baselines
            class_freq = class_counts[class_name] / total_count
            pos_baselines[class_name] = class_priors[class_name]
            neg_baselines[class_name] = (1 - class_priors[class_name])
            precision_baselines[class_name] = class_freq

        return pos_baselines, neg_baselines, precision_baselines

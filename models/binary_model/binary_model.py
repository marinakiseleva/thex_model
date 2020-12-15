"""
Binary Model

Computes probability of each class entirely independently using optimized binary classifiers. Almost the same as Independent Model, except the probabilities are not normalized across classes. Strictly concerned with binary probability of each class.

"""

import numpy as np

from mainmodel.mainmodel import MainModel
from classifiers.binary.optbinary import OptimalBinaryClassifier


class BinaryModel(MainModel):

    def __init__(self, **data_args):
        """
        Initialize Independent Model - call init method of parent class
        """
        self.name = "Binary Classifiers"
        super(BinaryModel, self).__init__(**data_args)

    def get_num_classes(self):
        """
        Overrides parent function
        """
        return 2

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

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            probabilities[class_name] = self.models[
                class_name].get_class_probability(x, self.normalize)
            MIN_PROB = 0.0001  # Min probability to avoid overflow
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = MIN_PROB
                print(self.name + " NULL probability for " + class_name)

            if probabilities[class_name] < MIN_PROB:
                probabilities[class_name] = MIN_PROB

        return probabilities

    def is_true_positive(self, row, eval_class_index):
        """
        Determines if the prediction is a true positive for the class eval_class
        :param row: Row of probabilities in order of self.class_labels, and last column is label, as list.
        :param eval_class_index: index of class for which to evaluate on. So if this is the name of the class with the max prob, return True, otherwise, False.
        """
        pred_true = row[eval_class_index] > 0.5

        label_index = len(self.class_labels)
        labels = row[label_index]
        eval_class_name = self.class_labels[eval_class_index]
        is_class = self.is_class(eval_class_name, labels)

        # prob >0.5, and this is the true class
        return is_class and pred_true

    def get_true_pos_prob(self, row, class_index, max_class_prob):
        """
        Get true positive probability (need this to overwrite in Binary)
        """
        return row[class_index]

    def compute_baselines(self, class_counts, class_labels, y, class_priors=None):
        """
        Get random classifier baselines for completeness and purity
        Overwritten so that class priors are binary (1/2) instead of proportional to # of classes
        """
        comp_baselines = {}
        purity_baselines = {}
        total_count = y.shape[0]
        for class_name in class_labels:
            if class_priors is not None:
                class_rate = class_counts[class_name] / total_count
            else:
                class_rate = 1 / 2

            # Compute baselines
            TP = class_counts[class_name] * class_rate
            FP = total_count - class_counts[class_name]
            purity_baselines[class_name] = TP / (TP + FP)
            comp_baselines[class_name] = class_rate

        return comp_baselines, purity_baselines

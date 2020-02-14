"""
Multiclass Model


Creates single multiclass classifier that compares all classes at once. Assumes all classes are independent and are not related by any class hierarchy. 

"""

import numpy as np
import pandas as pd

from mainmodel.mainmodel import MainModel
from classifiers.multi.optmulti import OptimalMultiClassifier
import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS


class MultiModel(MainModel):

    def __init__(self, **data_args):
        """
        Initialize Multiclass Model - call init method of parent class
        """
        self.name = "Multiclass Model"
        super(MultiModel, self).__init__(**data_args)

    def train_model(self, X_train, y_train):
        """
        Train model using training data - single multiclass classifier
        """
        self.model = OptimalMultiClassifier(X_train, y_train, self.class_labels, self.nb)
        return self.model

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        # for class_index, class_name in enumerate(self.class_labels):
        probabilities = self.model.clf.get_class_probabilities(x)

        return probabilities

    def is_class(self, class_name, labels):
        """
        Boolean which returns True if class name is in the list of labels, and False otherwise.

        """
        if class_name in labels:
            return True
        else:
            return False

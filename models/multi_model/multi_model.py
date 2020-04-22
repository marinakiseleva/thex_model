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
        self.name = "Multiclass Classifier"
        super(MultiModel, self).__init__(**data_args)

    def train_model(self, X_train, y_train):
        """
        Train model using training data - single multiclass classifier
        """
        self.model = OptimalMultiClassifier(X=X_train,
                                            y=y_train,
                                            class_labels=self.class_labels,
                                            class_priors=self.class_priors,
                                            nb=self.nb,
                                            model_dir=self.dir)

        return self.model

    def get_class_probabilities(self, x, normalize=True):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        probabilities = self.model.clf.get_class_probabilities(x, normalize)

        return probabilities

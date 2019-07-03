import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from thex_data.data_consts import TARGET_LABEL


class SubClassifier(ABC):
    """
    Abstract class for classifiers within conditional probability classifier. A classifier is built for a set of siblings in the class hierarchy, thus it is just a SubClassifier. 
    """

    def __init__(self, classes, X, y):
        self.classes = classes
        self.classifier = self.init_classifier(X, y)

    def get_sample_weights(self, y):
        """
        Get weight of each sample (1/# of samples in class)
        :param y: Pandas DataFrame with TARGET_LABEL column
        """
        classes = y[TARGET_LABEL].unique()
        label_counts = {}
        for c in classes:
            label_counts[c] = y.loc[y[TARGET_LABEL] == c].shape[0]

        sample_weights = []
        for df_index, row in y.iterrows():
            class_count = label_counts[row[TARGET_LABEL]]
            sample_weights.append(1 / class_count)

        return np.array(sample_weights)

    @abstractmethod
    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Get the probability of each class for the feature row x. Return probabilities as list, in same order as self.classes
        :param x: 2D Numpy array of features for single row 
        """
        pass

import pandas as pd
import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from thex_data.data_consts import TARGET_LABEL


class KDEClassifier(BinaryClassifier):
    """
    Extension of abstract class for binary classifier.
    """

    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        self.model = self.get_best_model(X, y)
        return self.model

    def predict(self, X):
        """
        Get the probability of the positive class for each row in DataFrame X. Return probabilities as Numpy column.
        :param x: 2D Numpy array as column with probability of class per row
        """
        raise ValueError(
            "KDE Model overwrites get_all_class_probabilities so it does not need predict.")

    def get_best_model(self, X, y):
        """
        Get maximum likelihood estimated distribution by kernel density estimate of this class over all features
        :return: best fitting KDE
        """
        # Fit KDE to positive samples only.
        X = X.loc[y[TARGET_LABEL] == 1]
        y = y.loc[y[TARGET_LABEL] == 1]

        # Create grid to get optimal bandwidth
        range_bws = np.linspace(0.01, 10, 100)
        grid = {
            'bandwidth': range_bws,
            'kernel': ['gaussian'],
            'metric': ['euclidean']
        }

        clf_optimize = GridSearchCV(KernelDensity(), grid, iid=False, cv=3, n_jobs=-1)
        clf_optimize.fit(X)

        print("Optimal bandwidth")
        print(clf_optimize.best_params_)

        return clf_optimize.best_estimator_

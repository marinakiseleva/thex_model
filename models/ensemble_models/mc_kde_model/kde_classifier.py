import pandas as pd
import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


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
        # Create and fit training data to Tree
        labeled_samples = pd.concat([X, y], axis=1)
        sample_weights = self.get_sample_weights(labeled_samples)

        # Create grid to get optimal bandwidth
        range_bws = np.linspace(0.01, 2, 100)
        grid = {
            'bandwidth': range_bws,
            'kernel': ['gaussian'],
            'metric': ['euclidean']
        }
        clf_optimize = GridSearchCV(KernelDensity(), grid, iid=False, cv=3, n_jobs=-1)
        clf_optimize.fit(X)  # , y, sample_weight=sample_weights

        return clf_optimize.best_estimator_

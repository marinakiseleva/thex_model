import pandas as pd
import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class KDEClassifier(BinaryClassifier):
    """
    Extension of abstract class for binary classifier.
    """

    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        # Fit KDE to positive samples only.
        X_pos = X.loc[y[TARGET_LABEL] == 1]
        y_pos = y.loc[y[TARGET_LABEL] == 1]
        self.pos_model = self.get_best_model(X_pos, y_pos)

        # need negative model to normalize over
        X_neg = X.loc[y[TARGET_LABEL] == 0]
        y_neg = y.loc[y[TARGET_LABEL] == 0]
        self.neg_model = self.get_best_model(X_neg, y_neg)
        return self.pos_model

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
        # Create grid to get optimal bandwidth
        range_bws = np.linspace(0.01, 5, 200)
        grid = {
            'bandwidth': range_bws,
            'kernel': ['gaussian'],
            'metric': ['euclidean']
        }
        clf_optimize = GridSearchCV(KernelDensity(), grid,
                                    iid=False, cv=3, n_jobs=CPU_COUNT)
        clf_optimize.fit(X)
        # print("Optimal Parameters:")
        # print(clf_optimize.best_params_)

        clf = clf_optimize.best_estimator_

        # print("Total log-likelihood of training data: " + str(clf.score(X)))

        return clf

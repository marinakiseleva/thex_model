"""
Construct optimal binary classifier for the data passed in 

For right now: Just optimal KDE
"""

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class OptimalBinaryClassifier():

    def __init__(self, pos_class, X, y, priors=None):
        """
        Initialize binary classifier
        :param pos_class: class_name that corresponds to TARGET_LABEL == 1
        :param X: DataFrame of features
        :param y: DataFrame with TARGET_LABEL column, 1 if it has class, 0 otherwise
        :param priors: Priors to be used CURRENTLY NOT IMPLEMETNED
        """
        self.pos_class = pos_class
        self.classifier = self.init_classifier(X, y)
        self.classifier_type = 'KDE'

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

    def get_class_probability(self, x):
        """
        Get probability of this class for this sample x. Probability of class 1 = density(1) / (density(1) + density(0)). 
        :param x: Single row of features
        """
        pos_density = np.exp(self.pos_model.score_samples([x.values]))[0]
        neg_density = np.exp(self.neg_model.score_samples([x.values]))[0]
        # Normalize as binary probability first
        binary_prob = pos_density / (pos_density + neg_density)
        return binary_prob

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

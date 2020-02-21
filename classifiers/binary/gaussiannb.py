import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT, LOSS_FUNCTION


class GNBClassifier():
    """
    Gaussian Naive Bayes classifier
    """

    def __init__(self, X, y, sample_weights):
        """
        Init classifier through training
        """
        self.name = "Gaussian Naive Bayes"
        self.clf = self.train(X, y, sample_weights)

    def train(self, X, y, sample_weights):

        grid = {'var_smoothing': [0.001, 0.01, 0.1, 1, 10, 100]
                # 'criterion': ['entropy', 'gini'],
                # 'splitter': ['best', 'random'],
                }
        clf_optimize = GridSearchCV(
            estimator=GaussianNB(priors=[.5, .5]),
            param_grid=grid,
            scoring=LOSS_FUNCTION,
            cv=3,
            iid=True,
            n_jobs=CPU_COUNT)

        # Fit the random search model
        clf_optimize.fit(X.values, y.values)
        clf = clf_optimize.best_estimator_
        print(self.name + " optimal parameters:\n" + str(clf_optimize.best_params_))
        sys.stdout.flush()  # Print to output file
        return clf

    def get_class_probability(self, x):
        # Return probability of class at index 1 (==1, positive class)
        return self.clf.predict_proba([x.values])[0][1]

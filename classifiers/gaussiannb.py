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

        grid = {
            # 'criterion': ['entropy', 'gini'],
            # 'splitter': ['best', 'random'],
            'var_smoothing': [10**-4, 10**-2, 10**-1]
            # 'priors': [.5, .5]
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
        print("\nOptimal GaussianNB Parameters:")
        print(clf_optimize.best_params_)

        return clf

    def get_class_probability(self, x):
        # Return probability of class at index 1 (==1, positive class)
        return self.clf.predict_proba([x.values])[0][1]

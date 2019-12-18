
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


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
            'var_smoothing': [10**-9, 10**-6]
            # 'priors': [.5, .5]
        }
        clf_optimize = GridSearchCV(
            estimator=GaussianNB(),
            param_grid=grid,
            scoring='brier_score_loss',
            cv=3,
            iid=True)

        # Fit the random search model
        clf_optimize.fit(X, y, sample_weight=sample_weights)
        clf = clf_optimize.best_estimator_

        return clf

    def get_class_probability(self, x):
        # Return probability of class at index 1 (==1, positive class)
        return self.clf.predict_proba([x.values])[0][1]

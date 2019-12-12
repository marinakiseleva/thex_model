import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from classifiers.dtclassifier import DTClassifier
from sklearn.model_selection import GridSearchCV

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class ADAClassifier():
    """
    ADA Boosted Decision Tree
    """

    def __init__(self,  X, y, sample_weights, class_weights, base_clf, name):
        """
        Init classifier through training
        """
        self.name = name
        self.clf = self.train(base_clf, X, y, sample_weights, class_weights)

    def train(self, base_clf, X, y, sample_weights, class_weights):

        # Set the parameters by cross-validation

        grid = {'base_estimator': [base_clf],
                'algorithm': ['SAMME'],
                # 'probability': [True],
                'n_estimators': [50, 100],
                'learning_rate': [.1, 1, 10]
                }

        clf_optimize = GridSearchCV(
            estimator=AdaBoostClassifier(),
            param_grid=grid,
            scoring='brier_score_loss',
            cv=3,
            iid=True,
            n_jobs=CPU_COUNT)

        # Fit the random search model
        clf_optimize.fit(X, y, sample_weight=sample_weights)
        clf = clf_optimize.best_estimator_

        return clf

    def get_class_probability(self, x):
        """
        Return probability of class at index 1 (probability of 1, positive class)
        """
        print('prob ')
        print(self.clf.predict_proba([x.values]))
        return self.clf.predict_proba([x.values])[0][1]

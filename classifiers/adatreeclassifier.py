import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from classifiers.dtclassifier import DTClassifier
from sklearn.model_selection import GridSearchCV

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT, LOSS_FUNCTION


class ADAClassifier():
    """
    ADA Boosted Decision Tree
    """

    def __init__(self,  X, y, sample_weights, base_clf, name):
        """
        Init classifier through training
        """
        self.name = name
        self.clf = self.train(base_clf, X, y, sample_weights)

    def train(self, base_clf, X, y, sample_weights):
        print("Training ADA on " + str(base_clf.name))
        print("with X data ")
        print(str(X))
        print("y data ")
        print(str(y))

        # Set the parameters by cross-validation

        grid = {'n_estimators': [50, 100],
                'learning_rate': [.1, 1, 10]
                }

        clf_optimize = GridSearchCV(
            estimator=AdaBoostClassifier(base_estimator=base_clf, algorithm='SAMME'),
            param_grid=grid,
            scoring=LOSS_FUNCTION,
            cv=3,
            iid=True,
            n_jobs=CPU_COUNT)

        # Fit the random search model

        clf_optimize.fit(X.values, y.values.T[0], sample_weight=sample_weights)
        clf = clf_optimize.best_estimator_

        return clf

    def get_class_probability(self, x):
        """
        Return probability of class at index 1 (probability of 1, positive class)
        """
        return self.clf.predict_proba([x.values])[0][1]

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
        """
        Train on ADABoosted version of classifier 
        (note: grid search on ADA doesn't work)
        """
        a = AdaBoostClassifier(base_estimator=base_clf,
                               algorithm='SAMME.R',
                               n_estimators=100,
                               learning_rate=1,
                               )
        a.fit(X.values, y.values, sample_weight=sample_weights)

        return a

    def get_class_probability(self, x):
        """
        Return probability of class at index 1 (probability of 1, positive class)
        """
        return self.clf.predict_proba([x.values])[0][1]

from collections import OrderedDict
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT
from models.conditional_models.conditional_model.classifier import SubClassifier


class SubKDE(SubClassifier):

    def init_classifier(self, X, y):
        """
        Create KDE for each class in self.classes
        :param X: DataFrame of features
        :param y: DataFrame of numeric labels, each number corresponding to index in self.classes
        :return: map from class index in self.classes to KDE
        """
        self.kdes = {}
        # Create classifier for each class, present or not in sample
        for class_index, class_name in enumerate(self.classes):
            y_pos = y.loc[y[TARGET_LABEL] == class_index]
            X_pos = X.loc[y[TARGET_LABEL] == class_index]
            self.kdes[class_index] = self.get_best_model(X_pos, y_pos)

        return self.kdes

    def predict(self, x):
        """
        Return probabilities as list, in same order as self.classes
        :param x: 2D Numpy array of features for single row
        """
        probabilities = OrderedDict()
        for class_index, class_name in enumerate(self.classes):
            model = self.kdes[class_index]
            probabilities[class_index] = np.exp(model.score_samples(x))[0]

        sum_probabilities = sum(probabilities.values())
        if sum_probabilities == 0:
            probabilities = {k: 0 for k, v in probabilities.items()}
        else:
            probabilities = {k: v / sum_probabilities for k, v in probabilities.items()}
        return list(probabilities.values())

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
        clf = GridSearchCV(KernelDensity(), grid, iid=False, cv=3, n_jobs=CPU_COUNT)
        clf.fit(X)
        print("Total log-likelihood of training data: " + str(clf.score(X)))

        print("Best model params  " + str(clf.best_params_))
        return clf.best_estimator_


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class DTClassifier():
    """
    Decision tree classifier
    """

    def __init__(self, X, y, sample_weights, class_weights):
        """
        Init classifier through training
        """
        self.name = "Decision Tree"
        self.clf = self.train(X, y, sample_weights, class_weights)

    def train(self, X, y, sample_weights, class_weights):

        # grid = {'criterion': ['entropy', 'gini'],
        #         'splitter': ['best', 'random'],
        #         'max_depth': [20, 50, None],
        #         'min_samples_split': [2, 4, 8, 0.05],
        #         'min_samples_leaf': [1, 2, 4, 8],
        #         'min_weight_fraction_leaf': [0, 0.001, 0.01],
        #         'max_features': [0.3, 0.5, None],
        #         'class_weight': ['balanced', class_weights]
        #         }
        grid = {
            'class_weight': ['balanced', class_weights]
        }
        clf_optimize = GridSearchCV(
            estimator=DecisionTreeClassifier(),
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
        # Return probability of class at index 1 (==1, positive class)
        return self.clf.predict_proba([x.values])[0][1]

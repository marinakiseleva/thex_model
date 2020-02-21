import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT, LOSS_FUNCTION


class SVMClassifier():
    """
    Support vector machine classifier
    """

    def __init__(self, X, y, sample_weights, class_weights):
        """
        Init classifier through training
        """
        self.name = "SVM"
        self.clf = self.train(X, y, sample_weights, class_weights)

    def train(self, X, y, sample_weights, class_weights):

        # Set the parameters by cross-validation

        grid = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                # 'degree': [2, 3, 4, 5],
                'gamma': [1e-3, 1e-4, 'auto', 'scale'],
                'C': [1, 10, 100, 1000],
                # 'class_weight': ['balanced', class_weights]
                }

        clf_optimize = GridSearchCV(
            estimator=SVC(probability=True, class_weight='balanced'),
            param_grid=grid,
            scoring=LOSS_FUNCTION,
            cv=3,
            iid=True,
            n_jobs=CPU_COUNT)

        # Fit the random search model
        clf_optimize.fit(X.values, y.values, sample_weight=sample_weights)
        clf = clf_optimize.best_estimator_
        print(self.name + " optimal parameters:\n" + str(clf_optimize.best_params_))
        sys.stdout.flush()  # Print to output file
        return clf

    def get_class_probability(self, x):
        """
        Note: SVM returns 0 or 1, not probability. We access this 0 or 1 prediction as probability though.
        """
        return self.clf.predict([x.values])[0]

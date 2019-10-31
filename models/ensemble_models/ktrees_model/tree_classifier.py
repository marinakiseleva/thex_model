import pandas as pd
import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from thex_data.data_consts import CPU_COUNT, TARGET_LABEL


class TreeClassifier(BinaryClassifier):
    """
    Extension of abstract class for binary classifier.
    """

    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        self.model = self.get_best_model(X, y)

        X_pos = X.loc[y[TARGET_LABEL] == 1]
        X_neg = X.loc[y[TARGET_LABEL] == 0]

        self.pos_dist = self.get_model_dist(X_pos, "positive")
        self.neg_dist = self.get_model_dist(X_neg, "negative")

        return self.model

    def get_class_probability(self, x):
        """
        Get probability of this class for this sample x
        :param x: Single row of features
        """
        prob = self.model.predict_proba([x.values])[0][1]
        return prob

    def predict(self, X):
        """
        Get the probability of the positive class for each row in DataFrame X. Return probabilities as Numpy column.
        :param x: 2D Numpy array as column with probability of class per row
        """
        pos_predictions = self.model.predict_proba(X)[:, 1]
        col_predictions = np.array([pos_predictions]).T
        return col_predictions

    def get_best_model(self, X, y):
        """
        Use RandomizedSearchCV to compute best hyperparameters for the model, using passed in X and y
        :return: Tree with parameters corresponding to best performance, already fit to data
        """
        # Get weight of each sample by its class frequency
        labeled_samples = pd.concat([X, y], axis=1)
        sample_weights = self.get_sample_weights(labeled_samples)
        class_weights = self.get_class_weights(labeled_samples)

        grid = {'criterion': ['entropy', 'gini'],
                'splitter': ['best', 'random'],
                'max_depth': [20, 50, None],
                'min_samples_split': [2, 4, 8, 0.05],
                'min_samples_leaf': [1, 2, 4, 8],
                'min_weight_fraction_leaf': [0, 0.001, 0.01],
                'max_features': [0.3, 0.5, None],
                'class_weight': ['balanced', class_weights]
                }
        clf_optimize = GridSearchCV(
            estimator=DecisionTreeClassifier(), param_grid=grid, scoring='brier_score_loss', cv=3, iid=True, n_jobs=CPU_COUNT)

        # Fit the random search model
        clf_optimize.fit(X, y, sample_weight=sample_weights)
        clf = clf_optimize.best_estimator_

        # print("Tree brier_score_loss: " + str(clf_optimize.best_score_))
        # print("Best params: ")
        # print(clf_optimize.best_params_)
        # print("Feature importance: ")
        # print(sorted(zip(X.columns, clf.feature_importances_),
        #              key=lambda x: x[1], reverse=True)[0:5])

        return clf

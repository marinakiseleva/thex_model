from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


from models.conditional_model.classifier import SubClassifier
from thex_data.data_consts import TARGET_LABEL
# from thex_data.data_clean import convert_class_vectors, relabel


class SubKTree(SubClassifier):

    def init_classifier(self, X, y):
        """
        Create K Decision Trees, compare a class in self.classes to the rest. 
        :param X: DataFrame of features
        :param y: DataFrame of numeric labels, each number corresponding to index in self.classes
        :return: map from class index in self.classes to Tree
        """
        self.ktrees = {}
        # Create classifier for each class, present or not in sample
        for class_index, class_name in enumerate(self.classes):
            y_relabeled = y.copy()
            # Relabel samples with class_index as 1 and all other classes as 0
            for index, row in y.iterrows():
                y_relabeled.at[index, TARGET_LABEL] = 1 if row[
                    TARGET_LABEL] == class_index else 0
            self.ktrees[class_index] = self.get_best_model(X, y_relabeled)

        return self.ktrees

    def predict(self, x):
        """
        Return probabilities as list, in same order as self.classes
        :param x: 2D Numpy array of features for single row
        """
        probabilities = OrderedDict()
        for class_index, class_name in enumerate(self.classes):
            tree = self.ktrees[class_index]

            if tree is not None:
                class_probabilities = tree.predict_proba(x)
                # class_probabilities = [[prob class 0, prob class 1]]
                class_probability = class_probabilities[0][1]
            else:
                raise ValueError("Each class should have a tree.")

            probabilities[class_name] = class_probability

        return list(probabilities.values())

    def get_best_model(self, X, y):
        """
        Use GridSearchCV to compute best hyperparameters for the model, using passed in X and y
        :return: Tree with parameters corresponding to best performance, already fit to data
        """
        # Get weight of each sample by its class frequency
        sample_weights = self.get_sample_weights(y)

        grid = {'criterion': ['entropy', 'gini'],
                'splitter': ['best', 'random'],
                'max_depth': [50],
                'min_samples_split': [2, 4, 8, 0.05],
                'min_samples_leaf': [1, 3],
                'min_weight_fraction_leaf': [0, 0.001, 0.01],
                'max_features': [0.3, None],
                'class_weight': ['balanced']
                }
        clf_optimize = GridSearchCV(
            estimator=DecisionTreeClassifier(), param_grid=grid, scoring='brier_score_loss', cv=3, iid=True, n_jobs=12)

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

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from models.conditional_models.conditional_model.classifier import SubClassifier


class SubMCDTree(SubClassifier):

    def init_classifier(self, X, y):
        """
        Create Multiclass DecisionTreeClassifier to classify self.classes 
        :param X: DataFrame of features
        :param y: DataFrame of numeric labels, each number corresponding to index in self.classes
        :return: map from class index in self.classes to Tree
        """
        self.classifier = self.get_best_model(X, y)
        return self.classifier

    def predict(self, x):
        """
        Return probabilities as list, in same order as self.classes
        :param x: 2D Numpy array of features for single row
        """

        probabilities = self.classifier.predict_proba(x).tolist()[0]
        return probabilities

    def get_best_model(self, X, y):
        """
        Use GridSearchCV to compute best hyperparameters for the model, using passed in X and y
        :return: Tree with parameters corresponding to best performance, already fit to data
        """
        # Get weight of each sample by its class frequency
        class_weights = self.get_class_weights(y)
        sample_weights = self.get_sample_weights(y)
        grid = {'criterion': ['entropy', 'gini'],
                'splitter': ['best', 'random'],
                'max_depth': [20, 50, None],
                'min_samples_split': [2, 4, 0.05, 0.1],
                'min_samples_leaf': [1, 2, 4, 8],
                'min_weight_fraction_leaf': [0, 0.001, 0.01],
                'max_features': [0.3, 0.5, None],
                'class_weight': ['balanced', class_weights]  # class_weights
                }

        clf_optimize = GridSearchCV(
            estimator=DecisionTreeClassifier(),
            param_grid=grid,
            scoring='balanced_accuracy',
            cv=3,
            iid=True,
            n_jobs=-1
        )
        clf_optimize.fit(X, y, sample_weight=sample_weights)

        clf = clf_optimize.best_estimator_

        # print("Tree brier_score_loss: " + str(clf_optimize.best_score_))
        print("Best params: ")
        print(clf_optimize.best_params_)
        # print("Feature importance: ")
        # print(sorted(zip(X.columns, clf.feature_importances_),
        #              key=lambda x: x[1], reverse=True)[0:5])

        return clf

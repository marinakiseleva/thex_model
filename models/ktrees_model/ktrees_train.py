import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from thex_data.data_consts import TARGET_LABEL
from thex_data.data_clean import convert_class_vectors, relabel


class KTreesTrain:
    """
    Mixin for K-Trees model, training functionality
    """

    def get_sample_weights(self, labeled_samples):
        """
        Get weight of each sample (1/# of samples in class) and save in list with same order as labeled_samples
        :param labeled_samples: DataFrame of features and TARGET_LABEL, where TARGET_LABEL values are 0 or 1
        """
        classes = labeled_samples[TARGET_LABEL].unique()
        label_counts = {}
        for c in classes:
            label_counts[c] = labeled_samples.loc[
                labeled_samples[TARGET_LABEL] == c].shape[0]
        sample_weights = []
        for df_index, row in labeled_samples.iterrows():
            class_count = label_counts[row[TARGET_LABEL]]
            sample_weights.append(1 / class_count)
        return sample_weights

    def get_best_model(self, X, y):
        """
        Use RandomizedSearchCV to compute best hyperparameters for the model, using passed in X and y
        :return: Dictionary of tree parameters to best values {parameter: value}
        """
        # Create and fit training data to Tree
        labeled_samples = pd.concat([X, y], axis=1)
        sample_weights = self.get_sample_weights(labeled_samples)

        criterion = ['entropy', 'gini']
        splitter = ['best', 'random']
        max_depth = [50]
        min_samples_split = [2, 4, 8, 0.05]
        min_samples_leaf = [1, 3]
        min_weight_fraction_leaf = [0, 0.001, 0.01]

        grid = {'criterion': criterion,
                'splitter': splitter,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'min_weight_fraction_leaf': min_weight_fraction_leaf,
                'max_features': [0.3, None],
                'class_weight': ['balanced']
                }
        # basic_grid = {'criterion': ['gini'],
        #               'splitter': ['best'],
        #               'max_depth': [100],
        #               'min_samples_split': [3],
        #               'min_samples_leaf': [2],
        #               'min_weight_fraction_leaf': [0, 0.001],
        #               'max_features': [None, 0.2],
        #               'class_weight': ['balanced']
        #               }

        clf_optimize = GridSearchCV(
            estimator=DecisionTreeClassifier(), param_grid=grid, scoring='brier_score_loss', cv=3, iid=True, n_jobs=12)

        # Fit the random search model
        clf_optimize.fit(X, y, sample_weight=sample_weights)

        print("Tree brier_score_loss: " + str(clf_optimize.best_score_))
        clf = clf_optimize.best_estimator_
        print("Best params: ")
        print(clf_optimize.best_params_)
        print("Feature importance: ")
        print(sorted(zip(X.columns, clf.feature_importances_),
                     key=lambda x: x[1], reverse=True)[0:5])

        return clf

    def train(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        # Convert class labels to class vectors
        y_train_vectors = convert_class_vectors(self.y_train, self.class_labels)

        # Create classifier for each class, present or not in sample
        for class_index, class_name in enumerate(self.class_labels):
            # Labels for this tree
            y_train_labels = relabel(class_index, y_train_vectors)
            positive_count = y_train_labels.loc[
                y_train_labels[TARGET_LABEL] == 1].shape[0]
            if positive_count < 5:
                # Do not need to make tree for this class when there are less than X
                # positive samples
                self.models[class_name] = None
                continue
            print("\nClass " + class_name)
            clf = self.get_best_model(self.X_train, y_train_labels)

            self.models[class_name] = clf

        return self.models
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight


from thex_data.data_consts import TARGET_LABEL
from thex_data.data_clean import convert_str_to_list, relabel


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

    def get_class_weights(self, labeled_samples):
        """
        Get weight of each class
        :param labeled_samples: DataFrame of features and TARGET_LABEL, where TARGET_LABEL values are 0 or 1
        :return: dictionary with 0 and 1 as keys and values of class weight
        """
        class_weights = compute_class_weight(
            class_weight='balanced', classes=[0, 1], y=labeled_samples[TARGET_LABEL].values)
        return dict(enumerate(class_weights))

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

    def get_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels. 
        :param class_name: Positive class 
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def train(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        # Create classifier for each class, present or not in sample
        valid_classes = []
        for class_index, class_name in enumerate(self.class_labels):
            # Relabel for this tree
            y_relabeled = self.get_class_data(class_name, self.y_train)
            positive_count = y_relabeled.loc[y_relabeled[TARGET_LABEL] == 1].shape[0]
            if positive_count < 3:
                print("No model for " + class_name)
                continue

            print("\nK-Trees Class Model: " + class_name)
            clf = self.get_best_model(self.X_train, y_relabeled)
            self.models[class_name] = clf
            valid_classes.append(class_name)

        # Update class labels to only have classes for which we built models
        self.class_labels = valid_classes
        return self.models

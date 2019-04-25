import pandas as pd
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from thex_data.data_consts import TARGET_LABEL
from thex_data.data_clean import convert_class_vectors


class MCKDETrain:
    """
    Mixin for K-Trees model, training functionality
    """

    def relabel(self, class_index, class_vectors):
        """
        Relabel samples such that if they have a 1 in their class vector for class_index, they will be relabeled as 1; otherwise 0. Relabels TARGET_LABEL column of class_vectors
        :return: Pandas DataFrame with TARGET_LABEL column, filling with 1 if  class_vectors[TARGET_LABEL][class_index] is also 1, otherwise 0
        """
        labels = []
        for df_index, row in class_vectors.iterrows():
            class_vector = row[TARGET_LABEL]
            p = 1 if class_vector[class_index] == 1 else 0
            labels.append(p)
        class_vectors = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return class_vectors

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
        Get maximum likelihood estimated distribution by kernel density estimate of this class over all features
        :return: best fitting KDE
        """
        # Create and fit training data to Tree
        labeled_samples = pd.concat([X, y], axis=1)
        sample_weights = self.get_sample_weights(labeled_samples)

        # Create grid to search over bandwidth for
        range_bws = np.linspace(0.01, 2, 100)
        grid = {
            'bandwidth': range_bws,
            'kernel': ['gaussian'],
            'metric': ['euclidean']
        }
        clf_optimize = GridSearchCV(KernelDensity(), grid, cv=3)
        clf_optimize.fit(X, y, sample_weight=sample_weights)
        return clf_optimize.best_estimator_

    def train(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        # Convert class labels to class vectors
        y_train_vectors = convert_class_vectors(self.y_train, self.class_labels)

        self.models = {}

        # Create classifier for each class, present or not in sample
        for class_index, class_name in enumerate(self.class_labels):
            # Labels for this tree
            y_train_labels = self.relabel(class_index, y_train_vectors)
            positive_count = y_train_labels.loc[
                y_train_labels[TARGET_LABEL] == 1].shape[0]
            if positive_count < 5:
                # Do not need to make tree for this class when there are less than X
                # positive samples
                self.models[class_name] = None
                continue

            y_pos = y_train_labels.loc[y_train_labels[TARGET_LABEL] == 1]
            y_neg = y_train_labels.loc[y_train_labels[TARGET_LABEL] == 0]
            X_pos = self.X_train.loc[y_train_labels[TARGET_LABEL] == 1]
            X_neg = self.X_train.loc[y_train_labels[TARGET_LABEL] == 0]
            print("Class " + class_name)
            clf_pos = self.get_best_model(X_pos, y_pos)
            clf_neg = self.get_best_model(X_neg, y_neg)
            # print(clf.get_params())
            self.models[class_name] = [clf_pos, clf_neg]

        return self.models

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from models.base_model.base_model import BaseModel
from thex_data.data_consts import TARGET_LABEL, cat_code, code_cat
from thex_data.data_clean import convert_class_vectors


class KTreesModel(BaseModel):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia. 
    """

    def __init__(self, cols=None, col_matches=None, **data_args):
        self.name = "K-Trees Model"
        # do not use default label transformations; instead we will do it manually
        # in this class
        data_args['transform_labels'] = False
        # model will predict multiple classes per sample
        self.cols = cols
        self.col_matches = col_matches
        self.user_data_filters = data_args

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

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        # Convert class labels to class vectors
        y_train_vectors = convert_class_vectors(self.y_train, self.class_labels)

        self.ktrees = {}

        # Create classifier for each class, present or not in sample
        for class_index, class_name in enumerate(self.class_labels):
            # Labels for this tree
            y_train_labels = self.relabel(class_index, y_train_vectors)
            positive_count = y_train_labels.loc[
                y_train_labels[TARGET_LABEL] == 1].shape[0]
            if positive_count < 5:
                # Do not need to make tree for this class when there are less than X
                # positive samples
                self.ktrees[class_name] = None
                continue

            # Create and fit training data to Tree
            clf = DecisionTreeClassifier(
                criterion='gini', splitter='random', min_samples_split=2, min_samples_leaf=3, class_weight='balanced')
            labeled_samples = pd.concat([self.X_train, y_train_labels], axis=1)
            sample_weights = self.get_sample_weights(labeled_samples)
            clf.fit(self.X_train,  y_train_labels, sample_weight=sample_weights)
            self.ktrees[class_name] = clf

        return self.ktrees

    def test_model(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        num_samples = self.X_test.shape[0]
        default_response = np.matrix([0] * num_samples).T
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.ktrees[class_name]
            if tree is not None:
                tree_predictions = tree.predict(self.X_test)
                col_predictions = np.matrix(tree_predictions).T
                # Add predictions as column to predictions across classes
                m_predictions = np.append(m_predictions, col_predictions, axis=1)
            else:
                # Add column of 0s
                m_predictions = np.append(m_predictions, default_response, axis=1)

        return m_predictions

    def evaluate_model(self, test_on_train):
        class_recalls = self.get_mc_recall_scores()
        self.plot_performance(class_recalls, "KTrees Recall",
                              class_counts=None, ylabel="Recall")

        # Plot probability vs precision for each class
        X_accs = self.get_mc_probability_matrix()
        for class_index, class_name in enumerate(self.class_labels):
            perc_ranges, AP, TOTAL = self.get_mc_metrics_by_ranges(
                X_accs, class_name)
            if self.ktrees[class_name] is not None:
                acc = [AP[index] / T if T > 0 else 0 for index, T in enumerate(TOTAL)]
                self.plot_probability_ranges(perc_ranges, acc,
                                             'TP/Total', class_name, TOTAL)

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x). 
        :param x: Single row of features 
        :return: map from class_code to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.ktrees[class_name]
            if tree is not None:
                class_probabilities = tree.predict_proba([x.values])
                # class_probabilities = [[prob class 0, prob class 1]]
                class_probability = class_probabilities[0][1]
            else:
                class_probability = 0

            probabilities[class_name] = class_probability

        return probabilities

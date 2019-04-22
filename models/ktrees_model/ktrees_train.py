import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from thex_data.data_consts import TARGET_LABEL
from thex_data.data_clean import convert_class_vectors


class KTreesTrain:
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

    def train(self):
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

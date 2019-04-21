import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from models.base_model.base_model import BaseModel
from thex_data.data_consts import TARGET_LABEL, cat_code, code_cat
from thex_data.data_transform import convert_class_vectors


class KTreesModel(BaseModel):
    """
    Model that classifies using unique Kernel Density Estimates for distributions of each feature, of each class. 
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
        self.class_labels = list(cat_code.keys())

    def relabel(self, class_index, class_vectors):
        """
        Relabel samples such that if they have a 1 in their class vector for class_index, they will be relabeled as 1; otherwise 0. Relabels TARGET_LABEL column of self.y_train_vectors
        :return: Pandas DataFrame with TARGET_LABEL column, filling with 1 if  class_vectors[TARGET_LABEL][class_index] is also 1, otherwise 0
        """
        labels = []
        for df_index, row in class_vectors.iterrows():
            class_vector = row[TARGET_LABEL]
            p = 1 if class_vector[class_index] == 1 else 0
            labels.append(p)
        class_vectors = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return class_vectors

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        # Convert class labels to class vectors
        self.y_train_vectors = convert_class_vectors(self.y_train, self.class_labels)

        self.ktrees = {}

        # Create classifier for each class, present or not in sample
        for class_index, class_name in enumerate(self.class_labels):
            # Labels for this tree
            y_train_labels = self.relabel(class_index, self.y_train_vectors)
            positive_count = y_train_labels.loc[
                y_train_labels[TARGET_LABEL] == 1].shape[0]
            if positive_count == 0:
                # Do not need to make tree for this class when there are no positive
                # samples
                self.ktrees[class_name] = None
                continue
            # Create Tree
            print("Creating tree for " + class_name)
            clf = DecisionTreeClassifier(
                criterion='gini', splitter='best', class_weight='balanced')
            clf.fit(self.X_train,  y_train_labels)
            label_predictions = clf.predict(self.X_test)
            self.ktrees[class_name] = clf

        return self.ktrees

    def test_model(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
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

        self.predictions = pd.DataFrame(m_predictions)
        # print(self.predictions)
        print(self.predictions.shape)
        return m_predictions

    def evaluate_model(self, test_on_train):
        # Convert actual labels to class vectors for comparison
        self.y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)
        self.get_recall_scores()

    def get_recall_scores(self):
        """
        Get recall of each class
        """
        class_recalls = {label: 0 for label in self.class_labels}
        for class_index, class_name in enumerate(self.class_labels):
            class_recalls[class_name] = self.get_class_recall(class_index)
        print("\n\nRecall Performance\n")
        for key in class_recalls.keys():
            print(str(key) + " : " + str(round(class_recalls[key], 2)))

    def get_class_recall(self, class_index):
        """
        Get recall of class passed in using actual and predicted class vectors. Recall is TP/TP+FN
        """
        threshold = 0.4
        row_count = self.y_test_vectors.shape[0]
        TP = 0  # True positive count, predicted = actual = 1
        FN = 0  # False negative count, predicted = 0, actual = 1
        for sample_index in range(row_count - 1):

            # Compare this index of 2 class vectors
            predicted_class = self.predictions[sample_index, class_index]

            actual_classes = self.y_test_vectors.iloc[sample_index][TARGET_LABEL]
            actual_class = actual_classes[class_index]

            if actual_class >= threshold:
                if actual_class == predicted_class:
                    TP += 1
                elif actual_class != predicted_class:
                    FN += 1
        denominator = TP + FN
        class_recall = TP / (TP + FN) if denominator > 0 else 0
        return class_recall

    def get_class_probabilities(self, x):
        return self.calculate_class_probabilities(x)

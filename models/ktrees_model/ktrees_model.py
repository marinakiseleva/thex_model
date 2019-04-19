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
        data_args['hierarchical_model'] = True
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
                continue
            # Create Tree
            print("Creating tree for " + class_name)
            clf = DecisionTreeClassifier(
                criterion='gini', splitter='best', class_weight='balanced')
            clf.fit(self.X_train,  y_train_labels)
            label_predictions = clf.predict(self.X_test)
            self.ktrees[class_name] = clf

        self.tree_classes = self.ktrees.keys()
        return self.ktrees

    def test_model(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        """
        for class_name in self.tree_classes:
            tree = self.ktrees[class_name]
            tree_predictions = tree.predict(self.X_test)
            print(np.transpose(tree_predictions))

        return predicted_classes

    def get_class_probabilities(self, x):
        # return 0
        return self.calculate_class_probabilities(x)

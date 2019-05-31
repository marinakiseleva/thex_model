import numpy as np
import pandas as pd
from models.base_model_mc.mc_base_model import MCBaseModel
from models.network_model.network_train import NetworkTrain

from thex_data.data_consts import cat_code, class_to_subclass


class NetworkModel(MCBaseModel, NetworkTrain):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia.
    """

    def __init__(self, **data_args):
        self.name = "Neural Network Model"
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.networks = {}

    def set_class_labels(self, y):
        class_labels = []
        for parent_class, subclasses in class_to_subclass.items():
            class_level = self.class_levels[parent_class]
            subnet_classes = self.get_subnet_classes(
                subclasses, y, parent_class)
            class_labels += subnet_classes

        self.class_labels = list(set(class_labels))
        print("\n Class labels:")
        print(self.class_labels)

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.class_labels = self.set_class_labels(self.y_train)
        return self.train()

    def test_model(self):
        """
        Get class prediction for each sample.
        """
        raise ValueError("test_model not impelmented")

    def get_all_class_probabilities(self):
        """
        Get class probability for each sample in self.X_test
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        predictions = []
        for df_index, x in self.X_test.iterrows():
            prob_map = self.get_class_probabilities(
                np.array([x.values]))
            # Convert to list - order is find because map was created with
            # self.class_labels
            probabilites = list(prob_map.values())
            predictions.append(list(prob_map.values()))
        return np.array(predictions)

    def get_parent_prob(self, class_name, probabilities):
        """
        Recurse up through tree, getting parent prob until we find a valid one. For example, there may only be CC, II, II P in CC so we need to inherit the probability of CC.
        """
        if class_name == "TTypes":
            return 1
        elif self.get_parent(class_name) in probabilities:
            return probabilities[self.get_parent(class_name)]
        else:
            return self.get_parent_prob(self.get_parent(class_name), probabilities)

    def get_parent(self, class_name):
        """
        Get parent class name of this class in tree
        """
        for parent_class, subclasses in class_to_subclass.items():
            if class_name in subclasses:
                return parent_class

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x). Compute conditional probability of each class.
        :param x: Single row of features
        :return: map from class_names to probabilities
        """

        if isinstance(x, pd.Series):
            x = np.array([x.values])

        """Default each probability to 1, so that when we compute conditional probabilities, childrent inherit parent probs when there is no comparison"""
        probabilities = {c: 1 for c in self.class_labels}

        # Step 1 - Get probability of each class, by net
        for parent_class, subnet in self.networks.items():
            predictions = subnet.network.predict(x=x,  batch_size=1)
            predictions = list(predictions[0])
            for pred_index, pred_class_prob in enumerate(predictions):
                pred_class_name = subnet.classes[pred_index]
                probabilities[pred_class_name] = pred_class_prob

        # Step 2 - Compute conditional probabilities
        for current_level in range(max(self.class_levels.values())):
            for class_name, probability in probabilities.items():
                if self.class_levels[class_name] == current_level:
                    probabilities[
                        class_name] *= self.get_parent_prob(class_name, probabilities)
        return probabilities

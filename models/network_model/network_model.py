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
        self.class_labels = class_labels

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
                np.array([x.values]), "root", None)
            # Fill undefined values
            prob_map = self.inherit_probabilities(prob_map)
            probabilites = list(prob_map.values())
            predictions.append(list(prob_map.values()))

        return np.array(predictions)

    def inherit_probabilities(self, probabilities):
        """
        If a class has no siblings, assign it the probability of its parent (since it has no subnet). 
        """
        # 1. Collect all classes across all networks
        net_classes = []
        for net in self.networks.values():
            net_classes += net.classes

        # 2. Find all class names that are not in any network
        missing_classes = set(net_classes) ^ set(self.class_labels)

        # 3. Fill this missing class probs w/ parent probs
        for class_name in missing_classes:
            probabilities[class_name] = probabilities[self.get_parent(class_name)]

        return probabilities

    def get_parent(self, class_name):
        """
        Get parent class name of this class in tree
        """
        for parent_class, subclasses in class_to_subclass.items():
            if class_name in subclasses:
                return parent_class

    def get_class_probabilities(self, x, subnet="root", probabilities=None):
        """
        Calculates probability of each transient class for the single test data point (x). Recurse over levels of the hierarchy, computing conditional probability of each class. 
        :param x: Single row of features 
        :param subnet: Current SubNetwork to run test point through. Start at root.
        :return: map from class_names to probabilities
        """
        # Set defaults for initial run.
        if subnet == "root":
            # Start from root (of class hierarchy) network
            subnet = self.networks[self.tree.root]
        if probabilities is None:
            probabilities = {c: 0 for c in self.class_labels}

        if isinstance(x, pd.Series):
            x = np.array([x.values])

        predictions = subnet.network.predict(x=x,  batch_size=1, verbose=0)
        predictions = list(predictions[0])

        # Update probabilities
        for pred_index, pred_class_prob in enumerate(predictions):
            pred_class_name = subnet.classes[pred_index]
            """ Multiply this class probability by its parents to get conditional probability. If no parent exists, just assign this probability (top of tree). """
            parent_class = self.get_parent(pred_class_name)
            if parent_class in probabilities:
                pred_class_prob *= probabilities[parent_class]
            probabilities[pred_class_name] = pred_class_prob

        for next_class in subnet.classes:
            if next_class in self.networks:
                probabilities = self.get_class_probabilities(
                    x, self.networks[next_class], probabilities)
        return probabilities

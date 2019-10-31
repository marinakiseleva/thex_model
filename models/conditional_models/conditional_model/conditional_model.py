from abc import abstractmethod
import numpy as np
import pandas as pd

from thex_data.data_print import print_styled
from thex_data.data_consts import class_to_subclass, PRED_LABEL, TREE_ROOT


from models.base_model_mc.mc_base_model import MCBaseModel
from models.conditional_models.conditional_model.conditional_train import ConditionalTrain


class ConditionalModel(MCBaseModel, ConditionalTrain):
    """
    Conditional Probability classifier that is built of many classifiers at different levels of the hierarchy. Each SubClassifier compares different sets of siblings in the class hierarchy, and conditional probabilites are computed based on parent probabilities. 
    """

    def __init__(self, **data_args):
        self.name = "Conditional Probability Model"
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.subclassifiers = {}

    @abstractmethod
    def create_classifier(self, classes, X, y):
        pass

    def train_model(self):
        if self.class_labels is None:
            self.class_labels = self.set_class_labels(self.y_train)
        return self.train()

    def get_all_class_probabilities(self, normalized=True):
        """
        Get class probability for each sample in self.X_test
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        class_probabilities = []
        for df_index, x in self.X_test.iterrows():
            prob_map = self.get_class_probabilities(
                np.array([x.values]))
            # Convert to list - order is fine because map was created with
            # self.class_labels
            class_probabilities.append(list(prob_map.values()))
        return np.array(class_probabilities)

    def get_class_probabilities(self, x, normalized=True):
        """
        Calculates probability of each transient class for the single test data point (x). Compute conditional probability of each class.
        :param x: 2D array of features, because predict functions expect 2D arrays - although it only contains 1 row. List of single Numpy array.
        :return: map from class_names to probabilities
        """

        if isinstance(x, pd.Series):
            x = np.array([x.values])

        """Default each probability to 1, so that when we compute conditional probabilities, childrent inherit parent probs when there is no comparison"""
        probabilities = {c: 1 for c in self.class_labels}

        # Step 1 - Get probability of each class
        for parent_class, subclassifier in self.subclassifiers.items():
            predictions = subclassifier.predict(x)
            for pred_index, pred_class_prob in enumerate(predictions):
                pred_class_name = subclassifier.classes[pred_index]
                if pred_class_name in probabilities:
                    probabilities[pred_class_name] = pred_class_prob

        # Step 2 - Compute conditional probabilities
        for current_level in range(max(self.class_levels.values())):
            for class_name, probability in probabilities.items():
                if self.class_levels[class_name] == current_level and class_name in probabilities:
                    probabilities[
                        class_name] *= self.get_parent_prob(class_name, probabilities)

        return probabilities

    def get_parent_prob(self, class_name, probabilities):
        """
        Recurse up through tree, getting parent prob until we find a valid one. For example, there may only be CC, II, II P in CC so we need to inherit the probability of CC.
        """
        if class_name == TREE_ROOT:
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

from abc import abstractmethod
import numpy as np
import pandas as pd
from models.base_model_mc.mc_base_model import MCBaseModel
from models.conditional_model.conditional_train import ConditionalTrain
from models.conditional_model.conditional_test import ConditionalTest
from thex_data.data_consts import class_to_subclass


class ConditionalModel(MCBaseModel, ConditionalTrain, ConditionalTest):
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

    def set_class_labels(self, y):
        class_labels = []
        for parent_class, subclasses in class_to_subclass.items():
            class_level = self.class_levels[parent_class]
            subnet_classes = self.get_subclf_classes(
                subclasses, y, parent_class)
            class_labels += subnet_classes

        self.class_labels = list(set(class_labels))
        print("\n Class labels:")
        print(self.class_labels)

    def train_model(self):
        if self.class_labels is None:
            self.class_labels = self.set_class_labels(self.y_train)
        return self.train()

    def test_model(self):
        """
        Get class prediction for each sample.
        """
        return self.test()

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

    def get_class_probabilities(self, x):
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
                probabilities[pred_class_name] = pred_class_prob

        # Step 2 - Compute conditional probabilities
        for current_level in range(max(self.class_levels.values())):
            for class_name, probability in probabilities.items():
                if self.class_levels[class_name] == current_level:
                    probabilities[
                        class_name] *= self.get_parent_prob(class_name, probabilities)
        return probabilities

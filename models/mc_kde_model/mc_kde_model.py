from models.base_model_mc.mc_base_model import MCBaseModel
from models.mc_kde_model.mc_kde_train import MCKDETrain
from models.mc_kde_model.mc_kde_test import MCKDETest

import numpy as np


class MCKDEModel(MCBaseModel, MCKDETrain, MCKDETest):
    """
    Multiclass Kernel Density Estimate (KDE) model. Creates KDE for each class, and computes probability by normalizing over class at same level in class hierarchy. 
    """

    def __init__(self, **data_args):
        self.name = "Multiclass KDE Model"
        # do not use default label transformations; instead we will do it manually
        # in this class
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.models = {}

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.set_class_labels(self.y_train)
        print("Classes:\n------------------\n")
        print(self.class_labels)
        return self.train()

    def test_model(self):
        """
        Get class prediction for each sample.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        return self.test()

    def get_all_class_probabilities(self):
        return self.test_probabilities()

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x). 
        :param x: Single row of features 
        :return: map from class_name to probabilities
        """
        densities = {}
        for class_index, class_name in enumerate(self.class_labels):
            model = self.models[class_name]
            densities[class_name] = np.exp(model.score_samples([x.values]))[0]
        sum_densities = sum(densities.values())
        probabilities = {k: v / sum_densities for k, v in densities.items()}
        return probabilities

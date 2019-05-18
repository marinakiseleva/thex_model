from models.base_model_mc.mc_base_model import MCBaseModel
from models.network_model.network_train import NetworkTrain
from models.network_model.network_test import NetworkTest

from thex_data.data_consts import cat_code


class NetworkModel(MCBaseModel, NetworkTrain, NetworkTest):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia.
    """

    def __init__(self, **data_args):
        self.name = "Neural Network Model"
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.models = {}

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.class_labels = self.get_mc_unique_classes(self.y_train)
        return self.train()

    def test_model(self):
        """
        Get class prediction for each sample.
        """
        raise ValueError("test_model not impelmented")

    def get_all_class_probabilities(self):
        return self.test_probabilities()

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x). 
        :param x: Single row of features 
        :return: map from class_code to probabilities
        """
        raise ValueError("get_class_probabilities not implemented")

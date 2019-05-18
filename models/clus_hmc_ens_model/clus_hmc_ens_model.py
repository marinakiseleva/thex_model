import sys

from models.base_model_mc.mc_base_model import MCBaseModel
from models.clus_hmc_ens_model.clus_hmc_ens_train import CLUSHMCENSTrain
from models.clus_hmc_ens_model.clus_hmc_ens_test import CLUSHMCENSTest


class CLUSHMCENS(MCBaseModel, CLUSHMCENSTrain, CLUSHMCENSTest):
    """
    Hierarchical Multi-Label Classifier based on predictive clustering tree (PCT) and using bagging. Implementation of CLUS-HMC-ENS outlined in Kocev, Dzeroski 2010. Splits using reduction in class-weighted variance.
    """

    def __init__(self, **data_args):
        self.name = "CLUS-HMC-ENS"
        data_args['transform_labels'] = False
        self.user_data_filters = data_args

    def train_model(self):
        """
        Builds hierarchy and tree, and fits training data to it.
        """
        if self.class_labels is None:
            self.class_labels = self.get_mc_unique_classes(self.y_train)

        labeled_samples, feature_value_pairs = self.setup_data()
        print("\n\nDone setting up data... ")
        self.root = self.train(labeled_samples, feature_value_pairs, remaining_depth=300)
        print("\n\nDone training...")

    def test_model(self):
        """
        Test model on self.y_train data.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        return self.test()

    def get_all_class_probabilities(self):
        return self.test()

    def get_class_probabilities(self, x):
        """
        Returns map of class name to probability. 
        """
        # probs is vector of probabilities in same order as self.class_labels
        probs = self.test_sample(self.root, x)
        class_probabilities = {}
        for class_name, probability in zip(self.class_labels, probs):
            class_probabilities[class_name] = probability
        return class_probabilities

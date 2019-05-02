import sys

from models.base_model.base_model import BaseModel
from models.clus_hmc_ens_model.clus_hmc_ens_train import CLUSHMCENSTrain
from models.clus_hmc_ens_model.clus_hmc_ens_test import CLUSHMCENSTest


class CLUSHMCENS(BaseModel, CLUSHMCENSTrain, CLUSHMCENSTest):
    """
    Hierarchical Multi-Label Classifier based on predictive clustering tree (PCT) and using bagging. Implementation of CLUS-HMC-ENS outlined in Kocev, Dzeroski 2010. Splits using reduction in class-weighted variance.
    """

    def __init__(self, cols=None, col_matches=None, **data_args):
        self.name = "CLUS-HMC-ENS"
        data_args['transform_labels'] = False
        self.cols = cols
        self.col_matches = col_matches
        self.user_data_filters = data_args
        self.class_labels = None

    def train_model(self):
        """
        Builds hierarchy and tree, and fits training data to it.
        """
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

    def evaluate_model(self, test_on_train):
        class_recalls, class_precisions = self.get_mc_metrics()
        self.plot_performance(class_recalls, "CLUS HMC ENS Recall",
                              class_counts=None, ylabel="Recall")
        self.plot_performance(class_precisions, "CLUS HMC ENS Precision",
                              class_counts=None, ylabel="Precision")

    def get_class_probabilities(self, x):
        print("\n\n need to implement get_class_probabilities for CLUS-HMC-ENS \n")
        sys.exit(-1)

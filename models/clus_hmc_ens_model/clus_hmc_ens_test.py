import numpy as np

from models.clus_hmc_ens_model.nodes import *


class CLUSHMCENSTest:
    """
    Mixin Class for CLUS-HMC-ENS model Testing functionality
    """

    def test(self):
        """
        Test model on self.y_train data.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        m_predictions = np.zeros((0, len(self.class_labels)))
        for index, test_point in self.X_test.iterrows():
            predicted_class_vector = self.test_sample(self.root, test_point)
            v = np.array([list(predicted_class_vector)])
            m_predictions = np.append(m_predictions, v, axis=0)

        return m_predictions

    def test_sample(self, node, testing_features):
        if type(node) == LeafNode:
            return node.guess

        # Select child node based on feature value in test_point
        test_value = testing_features[node.feature]
        if node.feature_value >= test_value:
            # Continue down to samples with this feature/value pair
            return self.test_sample(node.sample_greater, testing_features)
        else:
            return self.test_sample(node.sample_less, testing_features)

import numpy as np


class KTreesTest:
    """
    Mixin for K-Trees model, testing functionality
    """

    def test_probabilities(self):
        """
        Get class probability for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        # record independent probability per class
        num_samples = self.X_test.shape[0]
        default_response = np.array([[0] * num_samples]).T
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.models[class_name]
            pos_predictions = tree.predict_proba(self.X_test)[:, 1]
            col_predictions = np.array([pos_predictions]).T
            # Add probabilities of positive as column to predictions across classes
            m_predictions = np.append(m_predictions, col_predictions, axis=1)
        return m_predictions

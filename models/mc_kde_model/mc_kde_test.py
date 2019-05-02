import numpy as np


class MCKDETest:
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia. 
    """

    def test(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        num_samples = self.X_test.shape[0]
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            kde = self.models[class_name][0]  # Positive Model
            probabilities = np.exp(kde.score_samples(self.X_test.values))
            col_predictions = np.array([probabilities]).T
            col_predictions[col_predictions >= 0.5] = 1
            col_predictions[col_predictions < 0.5] = 0

            # Add predictions as column to predictions across classes
            m_predictions = np.append(m_predictions, col_predictions, axis=1)

        return m_predictions

    def test_probabilities(self):
        """
        Get class probability for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        num_samples = self.X_test.shape[0]
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            kde = self.models[class_name][0]  # Positive Model
            pos_probabilities = np.exp(kde.score_samples(self.X_test.values))
            probs = np.array([pos_probabilities]).T
            # Add probabilities of positive as column to predictions across classes
            m_predictions = np.append(m_predictions, probs, axis=1)

        return m_predictions

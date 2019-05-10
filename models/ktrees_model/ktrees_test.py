import numpy as np


class KTreesTest:
    """
    Mixin for K-Trees model, testing functionality
    """

    def test(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        num_samples = self.X_test.shape[0]
        default_response = np.array([[0] * num_samples]).T
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.models[class_name]
            tree_predictions = tree.predict(self.X_test)
            col_predictions = np.array([tree_predictions]).T
            # Add predictions as column to predictions across classes
            m_predictions = np.append(m_predictions, col_predictions, axis=1)

        return m_predictions

    def test_probabilities(self):
        """
        Get class probability for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        if self.test_level is None:
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
        else:
            # Normalize over probabilities in level
            class_densities = {}
            norm = np.zeros(self.X_test.shape[0])
            for class_index, class_name in enumerate(self.class_labels):
                tree = self.models[class_name]
                class_densities[class_name] = tree.predict_proba(self.X_test)[:, 1]
                norm = np.add(norm, class_densities[class_name])

            # Divide each density by normalization (sum of all densities)
            probabilities = np.zeros((self.X_test.shape[0], 0))
            for class_name in class_densities.keys():
                p = np.divide(class_densities[class_name], norm)
                probs = np.array([p]).T  # append probabilities as column
                probabilities = np.append(probabilities, probs, axis=1)
            return probabilities

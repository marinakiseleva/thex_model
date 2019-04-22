import numpy as np


class KTreesTest:
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia. 
    """

    def test(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        num_samples = self.X_test.shape[0]
        default_response = np.matrix([0] * num_samples).T
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.ktrees[class_name]
            if tree is not None:
                tree_predictions = tree.predict(self.X_test)
                col_predictions = np.matrix(tree_predictions).T
                # Add predictions as column to predictions across classes
                m_predictions = np.append(m_predictions, col_predictions, axis=1)
            else:
                # Add column of 0s
                m_predictions = np.append(m_predictions, default_response, axis=1)

        return m_predictions

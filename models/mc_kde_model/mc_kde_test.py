import numpy as np


class MCKDETest:
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia. 
    """

    def test(self):
        """
        Get class prediction for each sample.
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
        Get class probability for each sample.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        # Initialize all probabilites

        class_densities = {}
        for class_index, class_name in enumerate(self.class_labels):
            kde = self.models[class_name][0]  # Positive Model
            class_densities[class_name] = np.exp(
                kde.score_samples(self.X_test.values))

        # Compute normalization, the denominator
        norm_classes = self.get_norm_classes()
        norm = np.zeros(self.X_test.shape[0])
        for class_name in class_densities.keys():
            # Sum over all normalizations
            if class_name in norm_classes:
                norm = np.add(norm, class_densities[class_name])

        # Divide each probability by norm
        probabilities = np.zeros((self.X_test.shape[0], 0))
        for class_name in class_densities.keys():
            p = np.divide(class_densities[class_name], norm)
            probs = np.array([p]).T
            # Add probabilities as column
            probabilities = np.append(probabilities, probs, axis=1)

        return probabilities

    def get_norm_classes(self):
        """
        Get classes to normalize over (the sum of these class probabilities will be in the denominator of Bayes Theorem). By default, it will go over all classes. 
        """
        return self.class_labels

import numpy as np


class MCKDETest:
    """
    Testing Mixin for Multiclass Kernel Density Estimate testing code. Gets probabilities/predictions for samples.
    """

    def test(self):
        """
        Get class prediction for each sample. If probability >50% predict positive.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        class_densities = {}
        norm = np.zeros(self.X_test.shape[0])
        for class_index, class_name in enumerate(self.class_labels):
            kde = self.models[class_name][0]  # Positive Model
            class_densities[class_name] = np.exp(
                kde.score_samples(self.X_test.values))
            norm = np.add(norm, class_densities[class_name])

        # Divide each density by normalization (sum of all densities)
        predictions = np.zeros((self.X_test.shape[0], 0))
        for class_name in class_densities.keys():
            p = np.divide(class_densities[class_name], norm)
            col_predictions = np.array([p]).T  # append predictions as column

            col_predictions[col_predictions >= 0.5] = 1
            col_predictions[col_predictions < 0.5] = 0
            predictions = np.append(predictions, col_predictions, axis=1)

        return probabilities

    def test_probabilities(self):
        """
        Get class probability for each sample.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        class_densities = {}
        norm = np.zeros(self.X_test.shape[0])
        for class_index, class_name in enumerate(self.class_labels):
            kde = self.models[class_name][0]  # Positive Model
            class_densities[class_name] = np.exp(
                kde.score_samples(self.X_test.values))
            norm = np.add(norm, class_densities[class_name])

        # Divide each density by normalization (sum of all densities)
        probabilities = np.zeros((self.X_test.shape[0], 0))
        for class_name in class_densities.keys():
            p = np.divide(class_densities[class_name], norm)
            probs = np.array([p]).T  # append probabilities as column
            probabilities = np.append(probabilities, probs, axis=1)
        return probabilities

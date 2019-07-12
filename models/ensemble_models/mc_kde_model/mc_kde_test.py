import numpy as np
import pandas as pd
from thex_data.data_consts import PRED_LABEL


class MCKDETest:
    """
    Testing Mixin for Multiclass Kernel Density Estimate testing code. Gets probabilities/predictions for samples.
    """

    def test_probabilities(self):
        """
        Get class probability for each sample.
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        class_densities = {}
        norm = np.zeros(self.X_test.shape[0])
        for class_index, class_name in enumerate(self.class_labels):
            kde = self.models[class_name]  # Model
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

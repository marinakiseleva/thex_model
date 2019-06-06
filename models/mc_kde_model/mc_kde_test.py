import numpy as np
import pandas as pd
from thex_data.data_consts import PRED_LABEL


class MCKDETest:
    """
    Testing Mixin for Multiclass Kernel Density Estimate testing code. Gets probabilities/predictions for samples.
    """

    def test(self, keep_top_half=False):
        """
        Get class prediction for each sample. Predict class with max probability density.
        """
        # For diagnosing purposes - keep only top 1/2 probs
        if keep_top_half:
            unnormalized_max_probabilities = []
            for index, row in self.X_test.iterrows():
                # Save unnormalized probabilities
                unnormalized_probabilities = self.get_class_probabilities(
                    row, normalized=False)
                max_unnormalized_prob = max(unnormalized_probabilities.values())
                unnormalized_max_probabilities.append(max_unnormalized_prob)
            probs = np.array(unnormalized_max_probabilities)
            keep_indices = np.argwhere(probs > np.average(probs)).transpose()[0].tolist()

            # Get max class for those indices, and filter down self.X_test and
            # self.y_test to have same rows
            self.X_test = self.X_test.loc[self.X_test.index.isin(keep_indices)]
            self.y_test = self.y_test.loc[self.y_test.index.isin(keep_indices)]

        predictions = []
        for index, row in self.X_test.iterrows():
            probabilities = self.get_class_probabilities(row)
            max_prob_class = max(probabilities, key=probabilities.get)
            predictions.append(max_prob_class)
        predicted_classes = pd.DataFrame(predictions, columns=[PRED_LABEL])
        return predicted_classes

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

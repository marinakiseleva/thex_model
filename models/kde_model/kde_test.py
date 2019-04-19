from math import isnan
import pandas as pd
import numpy as np
from thex_data.data_consts import cat_code, UNKNOWN_LABEL, PRED_LABEL
"""
Logic for classifying testing data using the KDE Model
"""


class KDEModelTest:
    """
    Mixin Class for KDE Model Testing functionality, used in KDEModel
    """

    def test(self):
        """
        Tests model using established summaires & priors on test data. Returns class code predictions for these classes.
        """
        predictions = []
        for index, row in self.X_test.iterrows():
            predictions.append(self.test_sample(row))
        predicted_classes = pd.DataFrame(predictions, columns=[PRED_LABEL])
        return predicted_classes

    def test_sample(self, x):
        """
        Get probability for each class; return class that has maximum probability.
        :param x: Single row of features as datapoint
        """
        probabilities = self.calculate_class_probabilities(x)
        max_prob_class = max(probabilities, key=probabilities.get)

        return max_prob_class

    def calculate_class_probabilities(self, x):
        """
        Calculates probability of each transient class (the keys of summaries map), for the single test data point (x). Calculates probability by multiplying probability of each feature together. Returns map of class codes to probability.
        :param x: Single row of features as datapoint
        """
        probabilities = {}
        # Get probability density of each class, and add it to a running sum of
        # all probability densities
        for class_code, distribution in self.summaries.items():
            probabilities[class_code] = 1
            if self.naive:
                # Iterate through mean/stdev of each feature in features map
                for feature_name, dist in distribution.items():
                    test_value = x[feature_name]
                    if test_value is not None and not isnan(test_value):
                        prob_density = np.exp(dist.score_samples([[test_value]]))
                        # Multiply together probability of each feature
                        probabilities[class_code] *= prob_density
            else:
                # direct distribution
                prob_density = np.exp(distribution.score_samples([x.values]))
                # prob_density is array of just 1 value, for this 1 test point
                probabilities[class_code] *= prob_density[0]

            # Factor in prior
            probabilities[class_code] *= self.priors[class_code]

        # Normalize probabilities, to sum to 1
        sum_probabilities = sum(probabilities.values())
        probabilities = {k: v / sum_probabilities if sum_probabilities >
                         0 else 0 for k, v in probabilities.items()}

        # If probability < X% predict Unknown. X corresponds to precision
        unknown_prob = 1 if max(probabilities.values()) < self.threshold else 0
        probabilities[cat_code[UNKNOWN_LABEL]] = unknown_prob

        return probabilities

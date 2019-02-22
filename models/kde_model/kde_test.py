from math import isnan
import pandas as pd
import numpy as np

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
            max_class = self.test_sample(row)
            predictions.append(max_class)
        predicted_classes = pd.DataFrame(predictions, columns=['predicted_class'])
        return predicted_classes

    def test_sample(self, x):
        """
        Get probability for each class; return class that has maximum probability.
        :param x: Single row of features as datapoint
        """
        probabilities = self.calculate_class_probabilities(x)
        return max(probabilities, key=lambda k: probabilities[k])

    def calculate_class_probabilities(self, x, naive=False):
        """
        Calculates probability of each transient class (the keys of summaries map), for the test data point (x). Calculates probability by multiplying probability of each feature together. Returns map of class codes to probability.
        :param x: Single row of features as datapoint
        """
        probabilities = {}
        # Get probability density of each class, and add it to a running sum of
        # all probability densities
        for class_code, distribution in self.summaries.items():
            probabilities[class_code] = 1
            if naive:
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
                probabilities[class_code] *= prob_density

            # Factor in prior
            probabilities[class_code] *= self.priors[class_code]

        # Normalize probabilities, to sum to 1
        sum_probabilities = sum(probabilities.values())
        probabilities = {k: v / sum_probabilities for k, v in probabilities.items()}

        return probabilities

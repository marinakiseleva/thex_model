from math import isnan
import pandas as pd
"""
Logic for classifying testing data on Naive Bayes classifier
"""


def calculate_class_probabilities(summaries, priors, test_dp):
    """
    Calculates probability of each transient class (the keys of summaries map), for the test data point (test_dp). Calculates probability by multiplying probability of each feature together. Returns map of class codes to probability.
    """
    probabilities = {}
    # Get probability density of each class, and add it to a running sum of
    # all probability densities
    for transient_class, feature_mappings in summaries.items():
        probabilities[transient_class] = 1
        # Iterate through mean/stdev of each feature in features map
        for feature_name, f_dist in feature_mappings.items():
            test_value = test_dp[feature_name]
            dist = f_dist[0]
            if test_value is not None and not isnan(test_value):
                prob_density = dist(*f_dist[1]).pdf(test_value)

                # Multiply together probability of each feature
                probabilities[transient_class] *= prob_density

            # Factor in prior
            probabilities[transient_class] *= priors[transient_class]

    # Normalize probabilities, to sum to 1
    sum_probabilities = sum(probabilities.values())
    probabilities = {k: v / sum_probabilities for k, v in probabilities.items()}

    return probabilities


def test_sample(summaries, priors, test_point):
    """
    Run sample point through Naive Bayes distributions, and get probability for each class
    Returns: class that has maximum probability
    """
    probabilities = calculate_class_probabilities(summaries, priors, test_point)

    max_class = max(probabilities, key=lambda k: probabilities[k])
    return max_class


def test_set_samples(summaries, priors, testing_set):
    """
    Tests all samples in testing_set using Naive Bayes probabilities from summaries (created in training) and priors of each class
    """
    predictions = []
    for index, row in testing_set.iterrows():
        max_class = test_sample(summaries, priors, row)
        predictions.append(max_class)
    return predictions


def test_model(X_test, summaries, priors):
    """
    Tests model using established summaires & priors on test data (includes ttype)
    """
    predictions = test_set_samples(summaries, priors, X_test)
    predicted_classes = pd.DataFrame(predictions, columns=['predicted_class'])
    return predicted_classes

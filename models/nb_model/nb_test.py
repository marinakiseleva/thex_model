from math import isnan
import pandas as pd
import numpy as np
"""
Logic for classifying testing data using the Naive Bayes classifier
"""


def calculate_class_probabilities(summaries, priors, test_dp):
    """
    Calculates probability of each transient class (the keys of summaries map), for the test data point (test_dp). Calculates probability by multiplying probability of each feature together. Returns map of class codes to probability.
    :param summaries: Summaries computed by train_nb in nb_train.py
    :param priors: Priors computed by train_nb in nb_train.py
    :param test_point: Single row of features as datapoint
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
                # prob_density = dist(*f_dist[1]).pdf(test_value)
                parms = dist.get_params()
                prob_density = np.exp(dist.score_samples([[test_value]]))
                # Multiply together probability of each feature
                probabilities[transient_class] *= prob_density[0]

            # Factor in prior
            probabilities[transient_class] *= priors[transient_class]

    # Normalize probabilities, to sum to 1
    sum_probabilities = sum(probabilities.values())
    probabilities = {k: v / sum_probabilities for k, v in probabilities.items()}

    return probabilities


def test_sample(summaries, priors, test_point):
    """
    Run sample point through Naive Bayes distributions, and get probability for each class
    Returns class that has maximum probability.
    :param summaries: Summaries computed by train_nb in nb_train.py
    :param priors: Priors computed by train_nb in nb_train.py
    :param test_point: Single row of features as datapoint
    """
    probabilities = calculate_class_probabilities(summaries, priors, test_point)
    max_class = max(probabilities, key=lambda k: probabilities[k])
    return max_class


def test_nb(X_test, summaries, priors):
    """
    Tests model using established summaires & priors on test data. Returns class code predictions for these classes.
    :param X_test: Features of test data
    :param summaries: Summaries computed by train_nb in nb_train.py
    :param priors: Priors computed by train_nb in nb_train.py
    """
    # predictions = test_set_samples(summaries, priors, X_test)
    predictions = []
    for index, row in X_test.iterrows():
        max_class = test_sample(summaries, priors, row)
        predictions.append(max_class)
    predicted_classes = pd.DataFrame(predictions, columns=['predicted_class'])
    return predicted_classes

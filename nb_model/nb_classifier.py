import math
from thex_data.data_consts import code_cat, TARGET_LABEL
import scipy.stats as stats
# import numpy as np
"""
Logic for training and testing using Gaussian Naive Bayes
"""


def find_best_fitting_dist(data):
    """
    Finds best fitting distribution for this particular set of features (features of a single transient type)
    """
    distributions = [stats.norm]
    mles = []
    for distribution in distributions:
        # Fits data to distribution and returns MLEs of scale and loc
        # pars -> mu = loc, sigma = scale
        pars = distribution.fit(data)
        # negative loglikelihood: -sum(log pdf(x, theta), axis=0)
        mle = distribution.nnlf(pars, data)
        mles.append(mle)

    results = [(distribution.name, mle)
               for distribution, mle in zip(distributions, mles)]
    # Sorts smallest to largest -- smallest NNL is best
    best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
    # print('Best fit reached using {}, MLE value: {}'.format(
    #     best_fit[0].name, best_fit[1]))

    # Return best fitting distribution and parameters (loc and scale)
    return [best_fit[0], best_fit[0].fit(data)]


def summarize(df):
    """
    Summarizes features across df by getting mean and stdev across each column.
    """
    class_summaries = {}
    # get distribution of each feature
    for column_name in df:
        if column_name != TARGET_LABEL:
            col_values = df[column_name].dropna(axis=0)
            if len(col_values) > 0:
                class_summaries[column_name] = find_best_fitting_dist(col_values)

    return class_summaries


def separate_classes(train):
    """
    Separate by class (of unique transient types)
    Return map of {transient type : DataFrame of samples of that type}
    """
    transient_classes = list(train[TARGET_LABEL].unique())
    separated_classes = {}
    priors = {}  # Prior value of class, based on frequency
    total_count = train.shape[0]
    for transient in transient_classes:
        sub_df = train.loc[train[TARGET_LABEL] == transient]

        # SET PRIOR value, by frequency of class in total set
        # Uniform prior
        priors[transient] = 1 / len(transient_classes)
        class_count = sub_df.shape[0]
        # Inverted Frequency-based prior
        # priors[transient] = 1 - (class_count / total_count)

        # Set class value
        sub_df.drop([TARGET_LABEL], axis=1, inplace=True)
        separated_classes[transient] = sub_df
    print("Unique transient types: " + str(len(separated_classes)))

    return separated_classes, priors


def summarize_by_class(training_dataset):

    separated, priors = separate_classes(training_dataset)

    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries, priors


def calculate_class_probabilities(summaries, priors, test_dp):
    """
    Calculates probability of each transient class (the keys of summaries map), for the test data point (test_dp). Calculates probability by multiplying probability of each feature together. Returns map of class codes to probability.
    """
    probabilities = {}
    sum_probabilities = 0
    # Get probability density of each class, and add it to a running sum of
    # all probability densities
    for transient_class, feature_mappings in summaries.items():
        probabilities[transient_class] = 1
        # Iterate through mean/stdev of each feature in features map
        for feature_name, f_dist in feature_mappings.items():
            test_value = test_dp[feature_name]
            dist = f_dist[0]
            mu, sigma = f_dist[1]
            if test_value is not None and not math.isnan(test_value):
                prob_density = dist(mu, sigma).pdf(test_value)
                # Multiply together probability of each feature
                probabilities[transient_class] *= prob_density

            # Factor in prior
            probabilities[transient_class] *= priors[transient_class]
        # Keep track of total sum of probabilities for normalization
        sum_probabilities += probabilities[transient_class]

    # Normalize probabilities, to sum to 1
    for transient_class, probability in probabilities.items():
        if sum_probabilities == 0:
            probabilities[transient_class] = 0
        else:
            probabilities[transient_class] = probability / sum_probabilities

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
    Tests all samples in testing_set using Naive Bayes probabilities from summaries (created in summarize_by_class) and priors of each class
    """
    predictions = []
    for index, row in testing_set.iterrows():
        max_class = test_sample(summaries, priors, row)
        predictions.append(max_class)
    return predictions

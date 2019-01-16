import math
from thex_data.data_consts import TARGET_LABEL
"""
Logic for training and testing using Gaussian Naive Bayes
"""


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    numerator = sum([pow(x - avg, 2) for x in numbers])
    # denominator = float(len(numbers) - 1)
    variance = numerator / len(numbers)
    return math.sqrt(variance)


def summarize(df):
    """ 
    Summarizes features across df by getting mean and stdev across each column.
    Saves in map, mapping column name to list of values: col_name : [mean, stdev]
    """
    class_summaries = {}
    # get mean and stdev of each column in DataFrame
    for column_name in df:

        if column_name != TARGET_LABEL:
            col_values = df[column_name].dropna(axis=0)
            if len(col_values) > 0:
                # Take mean and stdev using only non-NULL values
                mean_col = mean(col_values)
                stdev_col = stdev(col_values)
                #  Map column name to mean and stdev of that column
                class_summaries[column_name] = [mean_col, stdev_col]

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


def calculate_probability_density(x, mean, stdev):
    """
    Using Gaussian PDF with passed in mean and stdev to find probability density of x
    """
    if stdev == 0:
        stdev = 0.2

    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    probability = (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    return probability


def calculate_class_probabilities(summaries, priors, test_dp):
    """
    Calculates probability of each transient class (the keys of summaries map), for the test data point (test_dp). Calculates probability by multiplying probability of each feature together.
    """
    probabilities = {}
    sum_probabilities = 0
    # Get probability density of each class, and add it to a running sum of all
    # probability densities
    for transient_class, feature_mappings in summaries.items():
        probabilities[transient_class] = 1

        # Iterate through mean/stdev of each feature in features map
        for feature_name, vals in feature_mappings.items():
            mean, stdev = vals
            test_value = test_dp[feature_name]
            if None not in (test_value, mean, stdev) and not math.isnan(test_value):
                prob_density = calculate_probability_density(test_value, mean, stdev)
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

    max_prob = 0
    max_class = 0
    for current_class, class_probability in probabilities.items():
        if class_probability > max_prob:
            max_prob = class_probability
            max_class = current_class

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

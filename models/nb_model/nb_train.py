from thex_data.data_consts import TARGET_LABEL, code_cat
from thex_data.data_print import print_priors
from models.nb_model.nb_performance import plot_dist_fit
import scipy.stats as stats
from sklearn.neighbors.kde import KernelDensity
import pandas as pd
import numpy as np

"""
Logic for training the Naive Bayes classifier
"""


def find_best_fitting_dist(data, feature=None, class_name=None):
    """
    Uses kernel density estimation to find the distribution of this data. Also plots data and distribution fit to it. Returns [distribution, parameters of distribution].
    :param data: set of values corresponding to feature and class
    :param feature: Name of feature/column this data corresponds to
    :param class_name: Name of class this data corresponds to
    """

    # Use Kernel Density

    # Set bandwidth of kernel
    range_vals = data.values.max() - data.values.min()
    min_val = data.values.min() - (range_vals / 5)
    max_val = data.values.max() + (range_vals / 5)
    bw = abs(max_val - min_val) / 100
    if bw == 0:
        bw = 0.1

    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    best_dist = kde.fit(np.matrix(data.values).T)
    best_params = kde.get_params()

    plot_dist_fit(data.values, kde, bw, "Kernel Distribution with bandwidth: %.6f\n for feature %s in class %s" % (
        bw, feature, class_name))

    # Return best fitting distribution and parameters (loc and scale)
    return [best_dist, best_params]


def summarize(data, class_name=None):
    """
    Estimate distribution of each feature in this data. Return mapping of {feature : [distribution, parameters of distribution]}
    :param data: DataFrame corresponding to all data of this class (class_name)
    :param class_name: Name of class this data corresponds to
    """
    class_summaries = {}
    # get distribution of each feature
    for feature in data:
        if feature != TARGET_LABEL:
            col_values = data[feature].dropna(axis=0)
            if len(col_values) > 0:
                class_summaries[feature] = find_best_fitting_dist(
                    col_values, feature, class_name)

    return class_summaries


def separate_classes(data):
    """
    Separate by class (of unique transient types)
    Return map of {class code : DataFrame of samples of that type}, and priors
    :param data: DataFrame of feature and labels
    """
    transient_classes = list(data[TARGET_LABEL].unique())
    separated_classes = {}
    priors = {}  # Prior value of class, based on frequency
    total_count = data.shape[0]
    for transient in transient_classes:
        trans_df = data.loc[data[TARGET_LABEL] == transient]

        # SET PRIOR value
        # Frequency of class in total set
        # priors[transient] = trans_df.shape[0] / total_count

        # Uniform prior
        priors[transient] = 1 / len(transient_classes)

        # Inverted Frequency-based prior
        # priors[transient] = 1 - (trans_df.shape[0] / total_count)

        # Set class value
        trans_df.drop([TARGET_LABEL], axis=1, inplace=True)
        separated_classes[transient] = trans_df

    # Make priors sum to 1
    priors = {k: round(v / sum(priors.values()), 6) for k, v in priors.items()}
    print_priors(priors)
    return separated_classes, priors


def train_nb(X_train, y_train):
    """
    Train Naive Bayes classifier on this training set
    :param X_train: Features of data
    :param y_train: Labels of data
    """
    training_dataset = pd.concat([X_train, y_train], axis=1)
    separated, priors = separate_classes(training_dataset)
    summaries = {}
    for class_code, instances in separated.items():
        summaries[class_code] = summarize(instances, code_cat[class_code])
    return summaries, priors

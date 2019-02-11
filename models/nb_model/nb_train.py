from thex_data.data_consts import TARGET_LABEL, code_cat
from nb_performance import plot_dist_fit
import scipy.stats as stats
from sklearn.neighbors.kde import KernelDensity
import pandas as pd
import numpy as np

"""
Logic for training the Naive Bayes classifier
"""


def find_best_fitting_dist(data, column_name=None, ttype=None):
    """
    Finds best fitting distribution for this particular set of features (features of a single transient type)
    """
    # distributions = [stats.norm]
    # , stats.beta, stats.gamma, stats.t, stats.gennorm, stats.alpha,  stats.arcsine,
    #                  stats.argus, stats.betaprime, stats.bradford, stats.burr, stats.burr12,
    #                  stats.cauchy, stats.chi, stats.chi2, stats.crystalball,
    #                  stats.dgamma, stats.dweibull, stats.laplace,
    #                  stats.skewnorm, stats.kappa4, stats.loglaplace]
    # mles = []
    # for distribution in distributions:
    #     # Fits data to distribution and returns MLEs of scale and loc
    #     pars = distribution.fit(data)
    #     # negative loglikelihood: -sum(log pdf(x, theta), axis=0)
    #     mle = distribution.nnlf(pars, data)
    #     mles.append(mle)

    # results = [(distribution.name, mle)
    #            for distribution, mle in zip(distributions, mles)]
    # # Sorts smallest to largest -- smallest NNL is best
    # ordered_dists = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
    # best_dist = ordered_dists[0]
    # best_params = ordered_dists[0].fit(data)

    # Use Kernel Density

    # Set bandwidth of kernel
    range_vals = data.values.max() - data.values.min()
    min_val = data.values.min() - (range_vals / 5)
    max_val = data.values.max() + (range_vals / 5)
    bw = (max_val - min_val) / 100

    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    best_dist = kde.fit(np.matrix(data.values).T)
    best_params = kde.get_params()

    plot_dist_fit(data.values, kde, bw, "Kernel Distribution with bandwidth: %.6f\n for feature %s in class %s" % (
        bw, column_name, ttype))

    # Return best fitting distribution and parameters (loc and scale)
    return [best_dist, best_params]


def summarize(df, ttype=None):
    """
    Summarizes features across df by getting mean and stdev across each column.
    """
    class_summaries = {}
    # get distribution of each feature
    for column_name in df:
        if column_name != TARGET_LABEL:
            col_values = df[column_name].dropna(axis=0)

            if len(col_values) > 0:
                class_summaries[column_name] = find_best_fitting_dist(
                    col_values, column_name, ttype)

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
        trans_df = train.loc[train[TARGET_LABEL] == transient]

        # SET PRIOR value
        # Frequency of class in total set
        priors[transient] = trans_df.shape[0] / total_count

        # Uniform prior
        # priors[transient] = 1 / len(transient_classes)

        # Inverted Frequency-based prior
        # priors[transient] = 1 - (trans_df.shape[0] / total_count)

        # Set class value
        trans_df.drop([TARGET_LABEL], axis=1, inplace=True)
        separated_classes[transient] = trans_df

    # Make priors sum to 1
    priors = {k: round(v / sum(priors.values()), 6) for k, v in priors.items()}

    print("\nPriors\n------------------")
    for k in priors.keys():
        print(code_cat[k] + " : " + str(priors[k]))
    return separated_classes, priors


def train_nb(X_train, y_train):
    """
    Train Naive Bayes classifier on training set
    """
    training_dataset = pd.concat([X_train, y_train], axis=1)
    separated, priors = separate_classes(training_dataset)
    summaries = {}
    for class_code, instances in separated.items():
        print("\n\n CLASS " + str(code_cat[class_code]) + "\n\n")
        summaries[class_code] = summarize(instances, code_cat[class_code])

    return summaries, priors

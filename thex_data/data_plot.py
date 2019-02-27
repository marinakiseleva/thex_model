import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
from .data_consts import code_cat, TARGET_LABEL, ROOT_DIR

FIG_WIDTH = 8
FIG_HEIGHT = 6

"""
data_plot contains helper functions to plot the data. This includes plotting the distribution of features, and the distributions of transient types over redshift.
"""


def plot_feature_distribution(df, feature, logged = True):
    """
    Plots the distribution of each transient type in df over 'feature'
    """

    unique_classes = list(df[TARGET_LABEL].unique())

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
    cm = plt.get_cmap('tab10')
    NUM_COLORS = len(unique_classes)
    ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    for class_code in unique_classes:
        values = df.loc[(df[TARGET_LABEL] == class_code)
                             & (df[feature].notnull())][feature].values
        vector_values = np.sort(np.matrix(values).T, axis=0)
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian', metric='euclidean')
        kde = kde.fit(vector_values) # Fit KDE to values
        pdf = kde.score_samples(vector_values) # Get PDF of values from KDE
        # ax.plot(vector_values, np.exp(pdf), label = code_cat[class_code])

        n, x, _ = ax.hist(vector_values, bins=np.linspace(0, .3, 50), alpha = 0.8,
                  label = code_cat[class_code])

    # if feature == "redshift":
    #     # Plot LSST-expected redshift distributions atop actual
    #     plot_lsst_distribution(ax)

    if logged:
        plt.yscale('log', nonposy='clip')
        ylabel = "Log Class Count"
    else:
        ylabel = "Class Count"
    plt.title("Transient Type Distributions over " + feature)
    plt.xlabel(feature)
    plt.ylabel(ylabel)
    plt.xlim(left=0)
    ax.legend(loc='best')
    plt.show()


def plot_lsst_distribution(ax):
    """
    Plot LSST Redshift distribution per class. Hard-coded based on collected
    expectations of redshift dists.
    """
    def plot_norm_class_dist(mu, sigma, label, ax):
        x = np.linspace(0, 1, 100)
        const = 1.0 / np.sqrt(2 * np.pi * (sigma**2))
        y = const * np.exp(-((x - mu)**2) / (2.0 * (sigma**2)))
        ax.plot(x, y, label = label)
    plot_norm_class_dist(mu=0.45, sigma=0.1, label="LSST Ia", ax=ax)

def count_classes(df):
    """
    Returns class codes and corresponding counts in DataFrame
    """
    class_sizes = pd.DataFrame(df.groupby(TARGET_LABEL).size())
    class_counts = {}
    for class_code, row in class_sizes.iterrows():
        class_counts[class_code] = row[0]
    return class_counts


def plot_class_hist(df):
    """
    Plots histogram of class sizes
    :param df: DataFrame with TARGET_LABEL column
    """

    class_counts = count_classes(df)
    class_names = [code_cat[c] for c in class_counts.keys()]
    class_indices = np.arange(len(class_names))

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
    plt.gcf().subplots_adjust(bottom=0.2)
    ax.bar(class_indices, list(class_counts.values()))
    plt.xticks(class_indices, class_names, fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    title = "Distribution of Transient Types in Data Sample"
    plt.title(title, fontsize=15)

    # Save to file
    replace_strs = ["\n", " ", ":", ".", ",", "/"]
    for r in replace_strs:
        title = title.replace(r, "_")
    plt.savefig(ROOT_DIR + "/output/classdistributions/" + title)
    plt.show()

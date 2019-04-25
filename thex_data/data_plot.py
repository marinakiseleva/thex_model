"""
data_plot
Functionality to plot data distributions and counts. This includes plotting the distribution of features, the distributions of transient types over redshift, and number of each transient type in dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors.kde import KernelDensity

from .data_consts import code_cat, TARGET_LABEL, ROOT_DIR

FIG_WIDTH = 8
FIG_HEIGHT = 6


def plot_feature_distribution(df, feature, transformed, logged=False):
    """
    Plots the distribution of each transient type in df over 'feature'
    """

    unique_classes = list(df[TARGET_LABEL].unique())

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
    cm = plt.get_cmap('tab20')
    NUM_COLORS = len(unique_classes)
    ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    # max_value = df[feature].max()
    max_value = 0.38
    for class_code in unique_classes:
        values = df.loc[(df[TARGET_LABEL] == class_code)
                        & (df[feature].notnull())][feature].values
        vector_values = np.sort(np.matrix(values).T, axis=0)
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian', metric='euclidean')
        kde = kde.fit(vector_values)  # Fit KDE to values
        pdf = kde.score_samples(vector_values)  # Get PDF of values from KDE

        if transformed:
            l = code_cat[class_code]
        else:
            l = class_code  # class code is actually name
        n, x, _ = ax.hist(vector_values, bins=np.linspace(
            0, max_value, 50), alpha=0.7, label=l)
        # ax.plot(vector_values, np.exp(pdf), label = l)

    # if feature == "redshift":
    #     # Plot LSST-expected redshift distributions atop actual
    #     plot_lsst_distribution(ax)

    if logged:
        plt.yscale('log', nonposy='clip')

    ylabel = "Class Count"
    plt.title("Transient Type Distributions over " + feature)
    plt.xlabel(feature)
    plt.ylabel(ylabel)
    plt.xlim(left=0, right=max_value)
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
        ax.plot(x, y, label=label)
    plot_norm_class_dist(mu=0.45, sigma=0.1, label="LSST Ia", ax=ax)


def count_classes(df):
    """
    Returns count of each distinct value in TARGET_LABEL of df
    :param df: Pandas DataFrame with TARGET_LABEL column
    :return class_counts: Map of {class_code : count, ...}
    """
    class_sizes = pd.DataFrame(df.groupby(TARGET_LABEL).size())
    class_counts = {}
    for class_code, row in class_sizes.iterrows():
        if class_code != '':
            class_counts[class_code] = int(row[0])
    return class_counts

from matplotlib.ticker import FormatStrFormatter


def plot_class_hist(df):
    """
    Plots histogram of class sizes
    :param df: DataFrame with TARGET_LABEL column
    """
    class_counts = count_classes(df)
    class_names = []

    for c in class_counts.keys():
        if c in code_cat:
            class_names.append(code_cat[c])
        else:
            class_names.append(str(c))
    num_classes = len(class_names)
    class_indices = np.arange(num_classes)

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
    plt.gcf().subplots_adjust(bottom=0.2)
    ax.bar(class_indices, list(class_counts.values()))
    plt.xticks(class_indices, class_names, fontsize=10)
    if num_classes > 5:
        plt.xticks(rotation=-90)

    if (max(class_counts.values()) - min(class_counts.values())) > 100:
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
        plt.tick_params(axis='y', which='minor')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.title("Distribution of Transient Types in Data Sample", fontsize=15)
    plt.show()

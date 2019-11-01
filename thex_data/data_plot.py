"""
data_plot
Functionality to plot data distributions and counts. This includes plotting the distribution of features, the distributions of transient types over redshift, and number of each transient type in dataset.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

from .data_consts import code_cat, TARGET_LABEL, ROOT_DIR, FIG_WIDTH, FIG_HEIGHT, DPI, cat_code
from .data_clean import convert_str_to_list


def plot_feature_distribution(df, feature, class_labels, class_counts):
    """
    Plots the distribution of each transient type in df over 'feature'
    :param df: DataFrame with both feature column and TARGET_LABEL column
    :param feature: Name of feature to plot distribution over
    :param class_labels: list of class names to show in legend
    :param class_counts: map fro class names to count
    """
    # Order classes  by count from largest to smallest so they plot well
    ordered_classes = []
    for class_name, class_count in sorted(class_counts.items(), key=lambda item: item[1],  reverse=True):
        ordered_classes.append(class_name)

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    cm = plt.get_cmap('tab20')
    NUM_COLORS = len(class_labels)
    if NUM_COLORS > 20:
        colors1 = plt.get_cmap('tab20b').colors
        colors2 = plt.get_cmap('tab20c').colors
        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
        cm = ListedColormap(colors)

    ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    max_value = df[feature].max()
    for class_name in ordered_classes:
        keep_indices = []
        for index, row in df.iterrows():
            classes = convert_str_to_list(row[TARGET_LABEL])
            if class_name in classes and row[feature] is not None:
                keep_indices.append(index)

        values = df.loc[keep_indices, :][feature].values
        vector_values = np.sort(np.matrix(values).T, axis=0)
        n, x, _ = ax.hist(vector_values, bins=np.linspace(
            0, max_value, 50), alpha=0.7, label=class_name)

    # if feature == "redshift":
    #     # Plot LSST-expected redshift distributions atop actual
    #     plot_lsst_distribution(ax)

    ylabel = "Class Count"
    title = "Transient Type Distributions over " + feature
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title, fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlim(left=0, right=max_value)
    plt.yscale('log', nonposy='clip')
    ax.legend(loc='best',  prop={'size': 5})
    plt.savefig(ROOT_DIR + "/output/" + title)
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


def plot_class_hist(df, target_is_name=False, class_counts=None):
    """
    Plots histogram of class sizes
    :param df: DataFrame with TARGET_LABEL column
    :param target_is_name: Boolean to use keys in count_classes dictionary as class labels.
    Can be True if TARGET_LABEL contains real class names and not codes.
    :param class_counts: Map from class name to counts, if pre-computed
    """
    if class_counts is None:
        class_counts = count_classes(df)
        class_names = []
        for c in class_counts.keys():
            if target_is_name:
                class_names.append(str(c))
            else:
                class_names.append(code_cat[c])
    else:
        class_names = list(class_counts.keys())

    num_classes = len(class_names)
    class_indices = np.arange(num_classes)

    f, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    # Plot data horizontally
    ax.barh(y=class_indices, width=list(class_counts.values()), height=0.7)

    plt.gcf().subplots_adjust(left=0.1)

    # Set logscale for range of values
    if (max(class_counts.values()) - min(class_counts.values())) > 100:
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
        plt.tick_params(axis='x', which='major', rotation=-90)

    ax.invert_yaxis()  # labels read top-to-bottom
    plt.yticks(class_indices, class_names, fontsize=8)
    ax.tick_params(axis='x', which='both', labelsize=8, rotation=-90)
    ax.tick_params(axis='x', which='minor', labelsize=4)

    plt.ylabel('Class', fontsize=12)
    plt.xlabel('Count', fontsize=12)

    title = "Distribution of Transient Types in Data Sample"
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT_DIR + "/output/" + title.replace(" ", "_"))
    plt.show()

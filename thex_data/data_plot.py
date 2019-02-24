import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

from .data_consts import code_cat, TARGET_LABEL, ROOT_DIR

FIG_WIDTH = 8
FIG_HEIGHT = 6

"""
data_plot contains helper functions to plot the data. This includes plotting the distribution of features, and the distributions of transient types over redshift.
"""


def plot_feature_distribution(df, feature):
    """
    Plots the distribution of each transient type in df over 'feature'
    """

    unique_ttypes = list(df[TARGET_LABEL].unique())

    fig, axs = plt.subplots(nrows=len(unique_ttypes), ncols=1, sharex=True,
                            sharey=True, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
    row_num = col_num = 0
    # find min and max values of this feature
    max_value = df[feature].max()
    min_value = df[feature].min()
    for ttype in unique_ttypes:
        values = list(df.loc[(df[TARGET_LABEL] == ttype)
                             & (df[feature].notnull())][feature])
        axs[row_num].hist(values, range=(min_value, max_value), bins=10)
        axs[row_num].set_title(code_cat[ttype])
        row_num += 1
        col_num = 0
    plt.suptitle("Transient Type Distributions over " + feature)
    plt.xlabel(feature)
    # plt.savefig("../output/feature_dist/" + feature)
    plt.show()


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

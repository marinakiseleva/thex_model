import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

from .data_consts import code_cat, TARGET_LABEL

"""
data_plot contains helper functions to plot the data. This includes plotting the distribution of features, and the distributions of transient types over redshift.
"""


def get_class_names(class_codes):
    """
    Convert class code numbers to corresponding strings (names) of classes of transient
    """
    tclasses_names = []
    for tclass in class_codes:
        tclasses_names.append(code_cat[tclass])
    return tclasses_names


def plot_feature_distribution(df, feature):
    """
    Plots the distribution of each transient type in df over 'feature'
    """
    rcParams['figure.figsize'] = 10, 10
    unique_ttypes = list(df[TARGET_LABEL].unique())

    fig, axs = plt.subplots(nrows=len(unique_ttypes), ncols=1, sharex=True, sharey=True)
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


def count_ttypes(df):
    """
    Returns transient type codes and corresponding counts of each type in dataframe df
    """
    ttype_counts = pd.DataFrame(df.groupby(TARGET_LABEL).size())
    types = []
    counts = []
    for ttype, row in ttype_counts.iterrows():
        t_count = row[0]
        if t_count > 0:
            counts.append(t_count)
            types.append(ttype)

    return types, counts


def map_counts_types(df):
    types, counts = count_ttypes(df)
    map_counts = {}
    for index, t in enumerate(types):
        map_counts[t] = counts[index]
    return map_counts


def plot_ttype_distribution(df):
    rcParams['figure.figsize'] = 10, 6
    plt.gcf().subplots_adjust(bottom=0.2)

    types, counts = count_ttypes(df)
    class_index = np.arange(len(types))
    plt.bar(class_index, counts)
    plt.xticks(class_index, get_class_names(types), fontsize=9, rotation=50)
    plt.xlabel('Transient Type', fontsize=12)
    plt.ylabel('Number of Galaxies', fontsize=12)
    plt.title("Distribution of Transient Types in Data Sample", fontsize=16)
    plt.show()

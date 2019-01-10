import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab
from pylab import rcParams
from math import ceil

from .data_maps import fnames, code_cat


def get_class_names(class_codes):
    """
    Convert class code numbers to corresponding strings (names) of classes of transient
    """
    tclasses_names = []
    for tclass in class_codes:
        class_name = code_cat[tclass]
        tclasses_names.append(class_name)
    return tclasses_names


def get_shapiro(values):
    """
    Performs Shapiro-Wilk test on values
    """
    stat, p = stats.shapiro(values)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


def plot_class_distribution(df, class_name='transient_type'):
    """
    Plots the distribution of a class over redshift
    """
    # unique_ttypes = list(df[class_name].unique())
    # for ttype in unique_ttypes:
    #     values = sorted(list(df.loc[df[class_name] == ttype]['redshift']))
    #     fit = stats.norm.pdf(x=values, loc=np.mean(values), scale=np.std(values))
    #     plt.plot(values, fit, '-o')
    #     # get_shapiro(values)
    # plt.legend(unique_ttypes, loc='best')
    # pylab.legend([code_cat[tt] for tt in unique_ttypes], loc='upper right')
    # plt.title("Transient Type Distribution")
    # plt.xlabel("Redshift")
    # plt.show()

    print("IN plot_class_distribution")
    unique_ttypes = list(df[class_name].unique())
    print(unique_ttypes)
    num_plot_rows = len(unique_ttypes)
    print(num_plot_rows)
    fig, axs = plt.subplots(nrows=num_plot_rows, ncols=1, sharex=True, sharey=True)
    row_num = col_num = 0
    for ttype in unique_ttypes:
        values = list(df.loc[(df[class_name] == ttype) & (df['redshift'] is not None)
                             # & (str(df['redshift']) != 'nan')
                             ]['redshift'])
        # print("new vals")
        # print(min(values))
        # print(max(values))
        print(ttype)
        print("Values")
        print(values)
        bin_vals = np.arange(min(values), max(values) + 0.1,
                             (max(values) - min(values)) / 20)
        print(bin_vals)
        # print(row_num)
        # print(col_num)
        axs[row_num].hist(values, bins=bin_vals)
        axs[row_num].set_title("Transient Type Distribution for " + str(ttype))
        axs[row_num].set_xlabel("Redshift")
        row_num += 1
        col_num = 0


def plot_feature_distribution(df):
    """
    Plots values of feature per sample to determine if they are normally distributed
    """

    features = list(df)
    features.remove('transient_type')
    print(features)
    for feature in features:
        values = sorted(list(df[feature]))
        fit = stats.norm.pdf(values, np.mean(values),
                             np.std(values))
        pylab.plot(values, fit, '-o')

    pylab.legend([fnames[f] if f in fnames else f for f in features], loc='upper left')
    pylab.title("Feature Distribution")
    pylab.show()


def count_ttypes(df):
    """
    Returns transient type codes and corresponding counts of each type in dataframe df
    """
    ttype_counts = pd.DataFrame(df.groupby('transient_type').size())
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

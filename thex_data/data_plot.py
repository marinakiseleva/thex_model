"""
data_plot
Functionality to plot data distributions and counts. This includes plotting the distribution of features, the distributions of transient types over redshift, and number of each transient type in dataset.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogLocator
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

from .data_consts import TARGET_LABEL, ROOT_DIR, FIG_WIDTH, FIG_HEIGHT, DPI, ORDERED_CLASSES, UNDEF_CLASS
import utilities.utilities as util


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def calculate_completeness(X, y, class_labels):
    """
    Get completeness of each class for each feature. Return as map from class name to list of completeness per feature.
    :param X: DataFrame of features
    :param y: DataFrame with TARGET_LABEL 
    :param class_labels: Class labels by which to calculate completeness
    """
    data = pd.concat([X, y], axis=1)
    features = list(X)
    completenesses = {}
    for class_name in class_labels:
        class_indices = []
        for index, row in data.iterrows():
            class_list = util.convert_str_to_list(row[TARGET_LABEL])
            if class_name in class_list:
                class_indices.append(index)
        class_data = data.loc[class_indices, :]

        features_completeness = []
        num_samples = class_data.shape[0]
        for feature in features:
            valid = class_data.dropna(subset=[feature])
            num_valid = valid.shape[0]
            features_completeness.append(num_valid / num_samples)
        completenesses[class_name] = features_completeness
    return completenesses


def visualize_completeness(model_dir, X, class_labels, data_completeness):
    """
    Plot completeness of dataset as heatmap. 
    :param model_dir: directory of model to save figure
    :param X: DataFrame of features
    :param class_labels: list of class names
    :param data_completeness: list in order of class names, which contains completeness per feature
    """
    features = list(X)
    features.sort()
    df = pd.DataFrame(data_completeness,
                      index=class_labels,
                      columns=features)

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    a = plt.pcolor(df, vmin=0, vmax=1, cmap='gist_heat')
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=-90)
    f.colorbar(a)

    plt.title("Completeness")
    util.display_and_save_plot(model_dir, "Completeness", None, f)


def plot_feature_distribution(model_dir, df, feature, class_labels, class_counts):
    """
    Plots the distribution of each transient type in df over 'feature'
    :param model_dir: directory of model to save figure
    :param df: DataFrame with both feature column and TARGET_LABEL column
    :param feature: Name of feature to plot distribution over
    :param class_labels: list of class names to show in legend
    :param class_counts: map from class names to count
    """

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
    for class_name in ORDERED_CLASSES:
        keep_indices = []
        for index, row in df.iterrows():
            classes = util.convert_str_to_list(row[TARGET_LABEL])
            if class_name in classes and row[feature] is not None:
                keep_indices.append(index)

        values = df.loc[keep_indices, :][feature].values
        vector_values = np.sort(np.array(values), axis=0)
        n, x, _ = ax.hist(vector_values, bins=np.linspace(
            0, max_value, 50), alpha=0.7, label=class_name)
    ylabel = "Class Count"

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(feature, fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlim(left=0, right=max_value)
    plt.yscale('log', nonposy='clip')
    ax.legend(loc='best',  prop={'size': 5})

    ax.set_title(feature)

    util.display_and_save_plot(
        model_dir, "Transient Type Distributions over " + feature)


def plot_class_hist(model_dir, class_names, counts):
    """
    Plots histogram of class sizes
    :param model_dir: directory of model to save figure
    :param class_counts: Map from class name to counts
    """

    class_indices = np.arange(len(class_names))
    f, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    # Plot data horizontally
    ax.barh(y=class_indices, width=counts, height=0.7)
    plt.gcf().subplots_adjust(left=0.1)
    # Set logscale for range of values
    if (max(counts) - min(counts)) > 100:
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
        plt.tick_params(axis='x', which='major', rotation=-90)

    plt.yticks(class_indices, class_names, fontsize=8)
    ax.tick_params(axis='x', which='both', labelsize=8, rotation=-90)
    ax.tick_params(axis='x', which='minor', labelsize=4)

    plt.ylabel('Class', fontsize=12)
    plt.xlabel('Count', fontsize=12)

    plt.title("Class Distribution", fontsize=12)
    plt.tight_layout()

    util.display_and_save_plot(
        model_dir, "Distribution of Transient Types in Data Sample")

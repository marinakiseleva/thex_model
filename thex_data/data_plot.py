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

from .data_consts import TARGET_LABEL, ROOT_DIR, FIG_WIDTH, FIG_HEIGHT, DPI, ORDERED_CLASSES, UNDEF_CLASS, ORDERED_MAGS
import utilities.utilities as util


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def get_ordered_features(features):
    """
    Order features if possible
    """

    ordered_features = []
    # Manually add first feature as it is not in defined dict.
    if 'FUV_mag' in features:
        ordered_features.append('FUV_mag')
    for mag in ORDERED_MAGS.keys():
        if mag in features:
            ordered_features.append(mag)

    # Order features by wavelength, if possible
    if len(ordered_features) == len(features):
        features = ordered_features
    return features


def calculate_completeness(X, y, class_labels):
    """
    Get completeness of each class for each feature. Return as map from class name to list of completeness per feature.
    :param X: DataFrame of features
    :param y: DataFrame with TARGET_LABEL 
    :param class_labels: Class labels by which to calculate completeness
    """
    data = pd.concat([X, y], axis=1)

    features = get_ordered_features(list(X))

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
    features = get_ordered_features(list(X))

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


def plot_feature_distribution(model_dir, df, feature, class_labels):
    """
    Plots the distribution of each transient type in df over 'feature'
    :param model_dir: directory of model to save figure
    :param df: DataFrame with both feature column and TARGET_LABEL column
    :param feature: Name of feature to plot distribution over
    :param class_labels: class labels
    """

    # Relabel DF
    label_col = df.columns.get_loc(TARGET_LABEL)
    for index, row in df.iterrows():
        classes = util.convert_str_to_list(row[TARGET_LABEL])
        for class_name in class_labels:
            if class_name in classes:
                df.iloc[index, label_col] = class_name
                break

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    max_value = 0.8  # df[feature].max()
    bins = np.linspace(0, max_value, 50)
    counts = []
    edges = []
    bars = []
    colors = plt.get_cmap('tab20').colors
    for index, class_name in enumerate(class_labels):
        class_rs = df[df[TARGET_LABEL] == class_name][feature].values
        c, e, b = ax.hist(x=class_rs,
                          bins=bins,
                          density=True,
                          color=colors[index],
                          label=class_name)
        counts.append(c)
        edges.append(e)
        bars.append(b)

    # Iterate over each bin
    it_bins = bars[0]
    for bin_index, value in enumerate(it_bins):
        bin_counts = []  # Count per class for this bin
        for class_count in counts:
            bin_counts.append(class_count[bin_index])

        # Sorted biggest to smallest, indices
        sorted_indices = np.flip(np.argsort(bin_counts))

        zorder = 0
        for sorted_index in sorted_indices:
            bars[sorted_index][bin_index].set_zorder(zorder)
            zorder += 1
    plt.xlabel(feature.capitalize())
    plt.ylabel("Normalized density")
    plt.legend()
    util.display_and_save_plot(model_dir, "Feature distribution")


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

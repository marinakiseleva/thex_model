"""
data_plot
Functionality to plot data distributions and counts. This includes plotting the distribution of features, the distributions of transient types over redshift, and number of each transient type in dataset.
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import matplotlib.pyplot as plt
from scipy.stats import norm
import os.path

from .data_consts import *
import utilities.utilities as util


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    plt.rcParams["font.family"] = "Times New Roman"


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

    if 'redshift' in features:
        ordered_features.append('redshift')

    return ordered_features


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
    for index, f in enumerate(features):
        if '_mag' in f:
            features[index] = f.replace("_mag", "")

    df = pd.DataFrame(data_completeness,
                      index=class_labels,
                      columns=features)

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    a = plt.pcolor(df, vmin=0, vmax=1, cmap='gist_heat')
    plt.yticks(ticks=np.arange(len(df.index)) + 0.6,
               labels=df.index,
               fontsize=TICK_S-2)
    plt.xticks(ticks=np.arange(len(df.columns)) + 0.5,
               labels=df.columns,
               fontsize=TICK_S + 2)
    f.colorbar(a)

    util.display_and_save_plot(model_dir, "Completeness", None, f)


def relabel_df(df, class_labels):
    """
    Helper function for plotting feature distributions. Relabels class label to be just a single label and not a list. The list of class_labels is assumed to be disjoint.
    """
    # Relabel DF
    label_col = df.columns.get_loc(TARGET_LABEL)
    for index, row in df.iterrows():
        classes = util.convert_str_to_list(row[TARGET_LABEL])
        for class_name in class_labels:
            if class_name in classes:
                df.iloc[index, label_col] = class_name
                break
    return df


def plot_feature_distribution(model_dir, df, feature, class_labels):
    """
    Plots the normal distribution of each transient type in df over 'feature'
    :param model_dir: directory of model to save figure
    :param df: DataFrame with both feature column and TARGET_LABEL column
    :param feature: Name of feature to plot distribution over
    :param class_labels: class labels
    """
    df = relabel_df(df, class_labels)

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    min_value = df[feature].min()
    max_value = df[feature].max()
    bins = np.linspace(0, max_value, 50)
    colors = plt.get_cmap('tab20').colors
    for index, class_name in enumerate(class_labels):
        class_values = df[df[TARGET_LABEL] == class_name][feature].values
        mean, std = norm.fit(class_values)
        x = np.linspace(min_value, max_value, 100)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, color=colors[index], label=class_name)
    plt.xlabel(feature.capitalize(), fontsize=LAB_S)
    plt.ylabel("Normalized density", fontsize=LAB_S)
    plt.legend()
    util.display_and_save_plot(model_dir, "feature_" + str(feature))


def plot_feature_hist(model_dir, df, feature, class_labels):
    """
    Plots the histogram of each transient type in df over 'feature'
    :param model_dir: directory of model to save figure
    :param df: DataFrame with both feature column and TARGET_LABEL column
    :param feature: Name of feature to plot distribution over
    :param class_labels: class labels
    """

    df = relabel_df(df, class_labels)

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    max_value = df[feature].max()
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
    plt.xlabel(feature.capitalize(), fontsize=LAB_S)
    plt.ylabel("Normalized density", fontsize=LAB_S)
    plt.legend()
    util.display_and_save_plot(model_dir, "Feature distribution")


def plot_df_feature_dists(model_dir, df1, df2, df1_name, df2_name, feature, class_labels):
    """
    Compare the feature distribution of one dataset vs another
    :param model_dir: directory of model to save figure 
    :param feature: Name of feature to plot distribution over 
    """

    df1 = relabel_df(df1, class_labels)
    df2 = relabel_df(df2, class_labels)

    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    min_value = min(df1[feature].min(), df2[feature].min())
    max_value = max(df1[feature].max(), df2[feature].max())

    x = np.linspace(min_value, max_value, 100)

    bins = np.linspace(0, max_value, 100)
    colors = plt.get_cmap('tab20').colors
    color_index = 0
    for class_name in class_labels:
        df1_vals = df1[df1[TARGET_LABEL] == class_name][feature].values
        mean1, std1 = norm.fit(df1_vals)
        df1_y = norm.pdf(x, mean1, std1)

        df2_vals = df2[df2[TARGET_LABEL] == class_name][feature].values
        mean2, std2 = norm.fit(df2_vals)
        df2_y = norm.pdf(x, mean2, std2)

        plt.plot(x, df1_y, color=colors[color_index], label=class_name + " " + df1_name)
        plt.plot(x, df2_y, color=colors[color_index + 1],
                 label=class_name + " " + df2_name)
        color_index += 2

    plt.xlabel(feature.capitalize(), fontsize=LAB_S)
    plt.ylabel("Normalized density", fontsize=LAB_S)
    plt.legend()


def plot_class_hist(model_dir, class_names, counts):
    """
    Plots histogram of class sizes
    :param model_dir: directory of model to save figure
    :param class_counts: Map from class name to counts
    """
    num_classes = len(class_names)
    f, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    # Plot data horizontally
    bar_width = 0.3
    if num_classes <= 5:
        tick_size = TICK_S + 1
        label_size = LAB_S + 2
        max_y = (bar_width * num_classes) - (bar_width / 2)
    else:
        tick_size = TICK_S + 3
        label_size = LAB_S + 2
        max_y = bar_width * (num_classes)
    class_indices = np.linspace(0, max_y, num_classes)
    ax.barh(y=class_indices, width=counts, height=bar_width, edgecolor='black')

    plt.gcf().subplots_adjust(left=0.1)

    ax.set_xscale('log')
    ax.set_xticks([50, 100, 500, 1000, 5000])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    plt.yticks(class_indices, class_names, fontsize=tick_size)
    ax.tick_params(axis='x', which='both',
                   labelsize=tick_size, rotation=-90)

    plt.xlabel('Host galaxies count (log scale)', fontsize=label_size)
    plt.tight_layout()
    util.display_and_save_plot(
        model_dir, "Distribution of Transient Types in Data Sample")

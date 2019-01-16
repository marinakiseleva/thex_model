import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import sys
sys.path.append("..")
from thex_data.data_consts import code_cat, TARGET_LABEL
from thex_data.data_plot import get_class_names


def combine_dfs(predicted_classes, actual_classes):
    """
    Combines predicted with actual classes in 1 DataFrame with new 'correct' column which has a 1 if the prediction matches the actual class, and 0 otherwise
    """
    df_compare = pd.concat([predicted_classes, actual_classes], axis=1)
    df_compare['correct'] = df_compare.apply(
        lambda row: 1 if row.predicted_class == row[TARGET_LABEL] else 0, axis=1)
    return df_compare


def get_percent_correct(df_compare):
    """
    Gets % of rows of dataframe that have correct column marked as 1. This column indicates if TARGET_LABEL == predicted_class
    """
    count_correct = df_compare[df_compare.correct == 1].shape[0]
    count_total = df_compare.shape[0]
    perc_correct = count_correct / count_total
    return perc_correct


def get_accuracy(predicted_classes, actual_classes):
    """
    Returns overall accuracy of Naive Bayes classifier
    """
    df_compare = combine_dfs(predicted_classes, actual_classes)
    perc_correct = get_percent_correct(df_compare)
    total_accuracy = round(perc_correct * 100, 4)
    return total_accuracy


def get_class_counts(df, classes):
    """
    Gets the count of each existing class in this dataframe
    """
    class_counts = []
    for tclass in classes:
        # Get count of this class
        class_count = df.loc[df[TARGET_LABEL] == tclass].shape[0]
        class_counts.append(class_count)
    return class_counts


def get_class_accuracies(df):
    """
    Get accuracy of each class separately
    """
    taccuracies = []  # Percent correctly predicted
    # List of corresponding class codes
    tclasses = list(df[TARGET_LABEL].unique())
    for tclass in tclasses:
        # filter df on this ttype
        df_ttype = df[df[TARGET_LABEL] == tclass]
        perc_correct = get_percent_correct(df_ttype)
        taccuracies.append(perc_correct)
    return taccuracies, tclasses


def filter_class_accuracies(taccuracies, tclasses):
    """
    Filter down to non-0 estimates to simplify graph
    """

    # Retain only accuracy values that are over 0.01, so to not clutter
    filter_taccuracies = []
    filter_classes = []
    for i in range(len(taccuracies)):
        if taccuracies[i] > 0.01:
            filter_taccuracies.append(taccuracies[i])
            filter_classes.append(tclasses[i])

    return filter_taccuracies, filter_classes


def prep_dataset_plot(predicted_classes, actual_classes):
    """
    Prep dataset to plot -- returns accuracy and the corresponding classes
    """
    # Get accuracy per class of transient
    df_compare = combine_dfs(predicted_classes, actual_classes)

    all_accuracies, all_transient_classes = get_class_accuracies(df_compare)
    # Filter down to non-0 accuracies
    accuracies, transient_classes = filter_class_accuracies(
        all_accuracies, all_transient_classes)
    low_accuracy_classes = len(all_transient_classes) - len(transient_classes)
    # Get row count of each class from df
    class_counts = get_class_counts(df_compare, transient_classes)

    print("Total classes: " + str(len(transient_classes)))
    print("0 accuracy class count: " + str(low_accuracy_classes))

    # Convert 2 lists to maps
    accs_by_class = {}
    for index, accuracy in enumerate(accuracies):
        tclass = transient_classes[index]
        accs_by_class[tclass] = accuracy
    return accs_by_class, class_counts


def normalize_datasets_compare(accs_by_class, accs_by_class2):
    """
    Rearranges/normalizes accuracies and transient classes betweeen 2 different sources so that they have the same transient classes 
    """
    # If accs_by_class2 is missing a key that accs_by_class has, add it and set it to 0
    for key in accs_by_class.keys():
        if key not in accs_by_class2.keys():
            accs_by_class2[key] = 0
    # vice versa
    for key in accs_by_class2.keys():
        if key not in accs_by_class.keys():
            accs_by_class[key] = 0
    return accs_by_class, accs_by_class2


def plot_compare_accuracy(predicted_classes, actual_classes, predicted_classes2, actual_classes2, data1_desc, data2_desc, plot_title):
    """
    Visualize accuracy per class with bar graph
    """
    # Prep dataset for plotting
    accs_by_class, class_counts = prep_dataset_plot(predicted_classes, actual_classes)
    accs_by_class2, class_counts2 = prep_dataset_plot(
        predicted_classes2, actual_classes2)

    accs_by_class, accs_by_class2 = normalize_datasets_compare(
        accs_by_class, accs_by_class2)

    # Get data into proper format for plotting: lists of x and y values
    transient_classes = list(accs_by_class.keys())  # same for both
    accuracies = [x for _, x in sorted(
        zip(list(accs_by_class.keys()), list(accs_by_class.values())))]
    accuracies2 = [x for _, x in sorted(
        zip(list(accs_by_class2.keys()), list(accs_by_class2.values())))]

    # Convert transient class codes into names
    tclasses_names = get_class_names(transient_classes)

    # Array will map to class names which are in same order as these indices
    class_index = np.arange(len(transient_classes)) + .5

    rcParams['figure.figsize'] = 10, 10

    # Plot dataset 1
    plt.barh(y=class_index, width=accuracies, height=.5, color='black')
    # Plot dataset2
    plt.barh(y=class_index + 0.5, width=accuracies2, height=.5, color='red')

    default_fontsize = 20
    plt.ylabel('Transient Class', fontsize=default_fontsize)
    plt.xlabel('Accuracy per Class', fontsize=default_fontsize)
    plt.yticks(class_index, tclasses_names, fontsize=default_fontsize)
    plt.xticks(list(np.linspace(0, 1, 11)), [
               str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=default_fontsize)
    plt.title(plot_title, fontsize=16)
    plt.gca().legend((data1_desc, data2_desc), fontsize=default_fontsize)

    cur_path = os.path.dirname(__file__)
    # plt.savefig(cur_path + "/../output/" + plot_title.replace(" ", "_"))

    plt.show()


def plot_class_accuracy(accuracies, transient_classes, plot_title="Accuracy per Transient Type"):
    rcParams['figure.figsize'] = 10, 10
    class_index = np.arange(len(transient_classes))
    plt.bar(class_index, accuracies)
    plt.xticks(class_index, transient_classes, fontsize=12, rotation=30)
    plt.yticks(list(np.linspace(0, 1, 11)), [
               str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=12)
    plt.title(plot_title, fontsize=15)
    plt.xlabel('Transient Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.show()


def compute_plot_class_accuracy(predicted_classes, actual_classes, plot_title):
    """
    Computes and Visualizes accuracy per class with bar graph
    """
    df_compare = combine_dfs(predicted_classes, actual_classes)
    # Get accuracy per class of transient
    all_accuracies, all_transient_classes = get_class_accuracies(df_compare)
    # Filter down to non-0 accuracies
    accuracies, transient_classes = filter_class_accuracies(
        all_accuracies, all_transient_classes)

    # Convert transient class codes into names
    tclasses_names = get_class_names(transient_classes)

    # Get row count of each class from df
    class_counts = get_class_counts(df_compare, transient_classes)

    # Generate array of length tclasses - class names will be assigned below
    # in yticks , in same order as these indices from tclasses_names
    class_index = np.arange(len(transient_classes))

    # Plot and save figure
    # rcParams['figure.figsize'] = 10, 10
    plt.bar(class_index, accuracies)

    cur_class = 0
    for xy in zip(class_index, accuracies):
        cur_count = class_counts[cur_class]  # Get class count of current class_index
        cur_class += 1
        plt.annotate(str(cur_count) + " total", xy=xy, textcoords='data',
                     ha='center', va='bottom', fontsize=12)
    plt.xlabel('Transient Class', fontsize=12)
    plt.ylabel('Accuracy per Class', fontsize=12)
    plt.xticks(class_index, tclasses_names, fontsize=12, rotation=30)
    plt.yticks(list(np.linspace(0, 1, 11)), [
               str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=12)
    plt.title(plot_title, fontsize=15)

    low_accuracy_classes = len(all_transient_classes) - len(transient_classes)
    print("Total classes: " + str(len(transient_classes)))
    print("0 accuracy class count: " + str(low_accuracy_classes))
    plt.show()

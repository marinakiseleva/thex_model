import itertools
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

from thex_data.data_consts import code_cat, TARGET_LABEL


def get_accuracy(predicted_classes, actual_classes):
    """
    Returns overall accuracy of Naive Bayes classifier
    """
    df_compare = combine_dfs(predicted_classes, actual_classes)
    perc_correct = get_percent_correct(df_compare)
    total_accuracy = round(perc_correct * 100, 4)
    return total_accuracy


def get_feature_importance(clf, train):
    cols = list(train.drop([TARGET_LABEL], axis=1))
    importances = clf.feature_importances_
    # print(importances)

    cols_importance = dict(zip(cols, importances))

    sorted_keys = sorted(cols_importance, key=cols_importance.get, reverse=True)
    for r in sorted_keys:
        if cols_importance[r] > 0:
            print(r, cols_importance[r])


def get_confusion_matrix(actual_classes, predicted_classes):
    cf_matrix = confusion_matrix(
        actual_classes, predicted_classes, labels=np.unique(actual_classes))
    return cf_matrix


def plot_confusion_matrix(actual_classes, predicted_classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = get_confusion_matrix(actual_classes, predicted_classes)
    classes = [code_cat[cc] for cc in np.unique(actual_classes)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    rcParams['figure.figsize'] = 8, 8
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


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


def get_percent_correct(df_compare):
    """
    Gets % of rows of dataframe that have correct column marked as 1. This column indicates if TARGET_LABEL == predicted_class
    """

    count_correct = df_compare[df_compare.correct == 1].shape[0]
    count_total = df_compare.shape[0]
    perc_correct = count_correct / count_total
    return perc_correct


def get_class_accuracies(df):
    """
    Get accuracy of each class separately
    """

    tclasses = list(df[TARGET_LABEL].unique())
    class_accuracies = {}
    for tclass in tclasses:
        df_ttype = df[df[TARGET_LABEL] == tclass]  # filter df on this ttype
        class_accuracies[tclass] = get_percent_correct(df_ttype)
    return collections.OrderedDict(sorted(class_accuracies.items()))


def combine_dfs(predicted_classes, actual_classes):
    """
    Combines predicted with actual classes in 1 DataFrame with new 'correct' column which has a 1 if the prediction matches the actual class, and 0 otherwise
    """

    PRED_LABEL = 'predicted_class'
    if type(predicted_classes) == list:
        predicted_classes = pd.DataFrame(predicted_classes, columns=[PRED_LABEL])
    if type(actual_classes) == list:
        actual_classes = pd.DataFrame(actual_classes, columns=[TARGET_LABEL])

    # Reset index in order to ensure concat works properly
    predicted_classes.reset_index(drop=True, inplace=True)
    actual_classes.reset_index(drop=True, inplace=True)

    df_compare = pd.concat([predicted_classes, actual_classes], axis=1)
    df_compare['correct'] = df_compare.apply(
        lambda row: 1 if row[PRED_LABEL] == row[TARGET_LABEL] else 0, axis=1)
    return df_compare


def plot_class_accuracy(class_accuracies, plot_title="Accuracy per Transient Type"):
    """
    Plots class accuracies as percentages only. Used for CV runs.
    :param class_accuracies: Mapping of classes to accuracies
    """
    transient_classes, accuracies = [], []
    for c in class_accuracies.keys():
        transient_classes.append(code_cat[c])
        accuracies.append(class_accuracies[c])
    rcParams['figure.figsize'] = 8, 8
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
    class_accuracies = get_class_accuracies(df_compare)

    # Convert transient class codes into names
    class_names = [code_cat[cc] for cc in class_accuracies.keys()]

    class_counts = get_class_counts(df_compare, class_accuracies.keys())

    # Generate array of length tclasses - class names will be assigned below
    # in yticks , in same order as these indices from tclasses_names
    class_index = np.arange(len(class_accuracies.keys()))

    # Plot and save figure
    rcParams['figure.figsize'] = 8, 8
    accuracies = [class_accuracies[c] for c in class_accuracies.keys()]
    plt.bar(class_index, accuracies)

    cur_class = 0
    for xy in zip(class_index, accuracies):
        cur_count = class_counts[cur_class]  # Get class count of current class_index
        cur_class += 1
        plt.annotate(str(cur_count) + " total", xy=xy, textcoords='data',
                     ha='center', va='bottom', fontsize=12)
    plt.xlabel('Transient Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(class_index, class_names, fontsize=12, rotation=30)
    plt.yticks(list(np.linspace(0, 1, 11)), [
               str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=12)
    plt.title(plot_title, fontsize=15)
    plt.show()

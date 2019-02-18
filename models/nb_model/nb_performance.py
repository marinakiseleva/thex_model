import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

from models.base_model.base_model_performance import *
from models.nb_model.nb_test import calculate_class_probabilities

from thex_data.data_consts import code_cat, TARGET_LABEL, ROOT_DIR
from thex_data.data_plot import get_class_names


def plot_dist_fit(data, kde, bandwidth, feature, title):
    """
    Plot distribution fitted to feature
    """
    plt.ioff()
    rcParams['figure.figsize'] = 6, 6
    n_bins = 20
    range_vals = data.max() - data.min()
    x_line = np.linspace(data.min(),
                         data.max(), 1000)
    data_vector = np.matrix(x_line).T
    pdf = kde.score_samples(data_vector)
    fig, ax1 = plt.subplots()

    ax1.hist(data, n_bins, fc='gray',  alpha=0.3)
    ax1.set_ylabel('Count', color='gray')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Kernel Density', color='blue')
    ax2.plot(x_line, np.exp(pdf), linewidth=3, alpha=0.5, label='bw=%.2f' % bandwidth)
    ax1.set_xlabel(feature)
    plt.title(title)
    replace_strs = ["\n", " ", ":", ".", ",", "/"]
    for r in replace_strs:
        title = title.replace(r, "_")
    plt.savefig(ROOT_DIR + "/output/kernel_fits/" + title)
    # plt.show()
    plt.close()
    plt.cla()

############################################################
# Plotting code for Naive Bayes ROC ########################


def get_rocs(test_data, actual_classes, summaries, priors):
    """
    Plot ROC curves for performance based on predictions, for each class
    :param test_data: Test data
    :param actual_classes: True labels for test set
    """

    actual_classes = actual_classes[TARGET_LABEL].reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    ttypes = list(set(actual_classes))
    f, ax = plt.subplots(len(ttypes), 3, figsize=(10, 10))
    for index, cur_class in enumerate(ttypes):
        get_roc(test_data, actual_classes, cur_class, summaries, priors, ax[index])
    plt.tight_layout()
    plt.show()


def get_roc(df, actual_classes, class_code, summaries, priors, ax):
    """
    Plot ROC Curve for this df (over 1 class)
    """
    # For each row in df, get probability of this class and whether it was right or wrong
    # pred_classes = []
    for index, row in df.iterrows():
        probabilities = calculate_class_probabilities(summaries, priors, row)
        prob = probabilities[class_code]
        # Save probability of this class for this row
        df.loc[index, 'probability'] = prob
        # Save whether or not this row IS this class
        actual_class = actual_classes.iloc[index]
        df.loc[index, 'is_class'] = True if (actual_class == class_code) else False

    prob_when_class = df.loc[df.is_class == True]['probability']
    prob_when_not_class = df.loc[df.is_class == False]['probability']
    ttype = code_cat[class_code]

    # Plot histogram of correct and incorrect probs
    ax[0] = plot_hist(prob_when_class, ax[0], "green")
    ax[0] = plot_hist(prob_when_not_class, ax[0], "red")
    ax[0].set_xlabel('P(class =' + ttype + ')', fontsize=12)
    ax[0].set_ylabel('Counts', fontsize=12)
    ax[0].set_title('Distribution', fontsize=15)
    ax[0].legend(["Type " + ttype, "NOT Type " + ttype])

    # Plot PDF for correct and incorrect distribution of probs
    x, y_prob_class = get_pdf(prob_when_class)
    ax[1] = plot_pdf(x, y_prob_class, ax[1], color="green")
    x, y_prob_when_not_class = get_pdf(prob_when_not_class)
    ax[1] = plot_pdf(x, y_prob_when_not_class, ax[1], color="red")
    ax[1].set_xlabel('P(class =' + ttype + ')', fontsize=12)
    ax[1].set_ylabel('PDF', fontsize=12)
    ax[1].set_title('Probability Distribution of ' + ttype, fontsize=15)
    ax[1].legend(["Type " + ttype, "NOT Type " + ttype])

    plot_roc(x, y_prob_class, y_prob_when_not_class, ax[2])


def plot_roc(x, y_pred_class, y_pred_not_class, ax):
    """
    Derive true positive rate (TPR) and false positive rate (FPR) for different thresholds of x. Create ROC curve based on that. 
    """
    # Total
    total_class_predicted = np.sum(y_pred_class)
    total_class_not_predicted = np.sum(y_pred_not_class)
    # Cumulative sum
    area_TP = 0  # Total area under correct curve
    area_FP = 0  # Total area under incorrect curve

    TP_rates = []  # True positive rates
    FP_rates = []  # False positive rates
    # Iteratre through all values of x
    for i in range(len(x)):
        if y_pred_class[i] > 0:
            area_TP += y_pred_class[len(x) - 1 - i]
            area_FP += y_pred_not_class[len(x) - 1 - i]
        # Calculate FPR and TPR for threshold x
        # Volume of false positives over total incorrects
        FPR = area_FP / total_class_not_predicted
        # Volume of true positives over total corrects
        TPR = area_TP / total_class_predicted
        TP_rates.append(TPR)
        FP_rates.append(FPR)
    # auc = Probability that it will guess true over false
    auc = np.sum(TP_rates) / 100  # 100= # of values for x
    # Plotting final ROC curve, FP against TP
    ax.plot(FP_rates, TP_rates)
    ax.plot(x, x, "--")  # baseline
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.grid()
    ax.legend(["AUC=%.3f" % auc])


def plot_hist(samples, ax, color):
    """
    Plots histogram of samples on axis ax
    """
    ax.hist(x=samples, bins=10, range=(0, 1),
            density=False, color=color, alpha=0.5)
    ax.set_xlim([0, 1])
    return ax


def get_pdf(probabilities):
    """
    Returns normal PDF values
    """
    samples = np.array(probabilities)
    mean = np.mean(samples)
    std = np.sqrt(np.var(samples))
    x = np.linspace(0, 1, num=100)
    const = 1.0 / np.sqrt(2 * np.pi * (std**2))
    y = const * np.exp(-((x - mean)**2) / (2.0 * (std**2)))
    return x, y


def plot_pdf(x, y, ax, color):
    """
    Plots the probability distribution function values of probabilities on ax
    """
    # Fit normal distribution to mean and std of data
    ax.plot(x, y, color, alpha=0.5)
    ax.set_xlim([0, 1])
    return ax


################################################################
# Plotting code for Naive Bayes Results ########################


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
    print("Classes with 0% accuracy: " + str(low_accuracy_classes))

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

    # cur_path = os.path.dirname(__file__)
    # plt.savefig(cur_path + "/../output/" + plot_title.replace(" ", "_"))

    plt.show()

"""
Plotting helpers for combining model outputs.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl

from models.binary_model.binary_model import BinaryModel
from models.ind_model.ind_model import OvAModel
from models.multi_model.multi_model import MultiModel
from mainmodel.helper_compute import *
from thex_data.data_consts import *
from utilities import utilities as thex_utils

import pickle


LSST_COLOR = "#80ccff"
THEX_COLOR = "#006600"


def load_prev_exp(exp_dir, expnum, model):

    # with open(pickle_dir + 'density_results.pickle', 'rb') as handle:
    #     density_results = pickle.load(handle)
    pickle_dir = exp_dir + expnum + "/"

    with open(pickle_dir + 'results.pickle', 'rb') as handle:
        results = pickle.load(handle)
    model.results = results

    with open(pickle_dir + 'y.pickle', 'rb') as handle:
        y = pickle.load(handle)
    model.y = y

    model.range_metrics = model.compute_probability_range_metrics(
        model.results, bin_size=0.2)
    model.range_metrics_10 = model.compute_probability_range_metrics(
        model.results, bin_size=0.1)
    return model

# Plotting formatting


def clean_class_name(class_name):
    pretty_class_name = class_name
    if UNDEF_CLASS in class_name:
        pretty_class_name = class_name.replace(UNDEF_CLASS, "")
        pretty_class_name = pretty_class_name + " (unspec.)"
    return pretty_class_name


def clean_class_names(class_names):
    """
    Update all references to Unspecified class to be class (unspec.)
    """
    new_class_names = []
    for class_name in class_names:
        pretty_class_name = clean_class_name(class_name)
        pretty_class_name.strip()
        new_class_names.append(pretty_class_name)
    return new_class_names


def prep_err_bars(intervals, metrics):
"""
Convert confidence intervals to specific values to be plotted, because xerr values are +/- sizes relative to the data:
"""
if intervals is None:
    return None
errs = [[], []]
for index, interval in enumerate(intervals):
    min_bar = interval[0]
    max_bar = interval[1]
    errs[0].append(metrics[index] - min_bar)
    errs[1].append(max_bar - metrics[index])
return errs


def get_perc_ticks():
    """
    Returns [0, 0.1, ..., 1], [10%, 30%, 50%, 70%, 90%]
    """
    indices = np.linspace(0, 1, 6)
    ticks = [str(int(i)) for i in indices * 100]
    return indices, ticks


def get_rates(class_name, model):
    """
    Get rates for this class under this model
    """
    true_positives, totals = model.range_metrics[class_name]
    pos_class_counts_per_range = np.array(model.class_positives[class_name])
    prob_rates = model.class_prob_rates[class_name]
    return prob_rates


def plot_model_rates(class_name, model, ax):
    """
    Plots rates for this model/class on axis, with annotations
    """
    true_positives, totals = model.range_metrics[class_name]
    pos_class_counts_per_range = np.array(model.class_positives[class_name])
    prob_rates = model.class_prob_rates[class_name]

    bins = np.arange(5)

    # color bars based on freq.
    # norm = plt.Normalize(0, max(totals))
    # colors = mpl.cm.Blues(norm(totals))

    ax.bar(bins, prob_rates, color=P_BAR_COLOR, edgecolor=BAR_EDGE_COLOR)
    ax.set_ylim(0, 1)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    index = 0

    for xy in zip(np.arange(5), prob_rates):
        # Get class count of current index
        count = str(totals[index])
        loc = list(xy)
        # lower annotation, so its not out of the plot for large bars
        if loc[1] > .9:
            xy = tuple([loc[0], loc[1] - .1])
        y_val = xy[1]
        ax.annotate(count, xy=xy, textcoords='data', ha='center',
                    va='bottom', fontsize=8)
        index += 1


def plot_rates_together(binary_model, ova_model, multi_model, indices=None):
    """
    Plot class versus probability rates of all three classifiers together
    :param indices: class indices to plot
    """
    rc('text', usetex=True)
    class_labels = ova_model.class_labels
    num_classes = len(ova_model.class_labels)
    if indices is not None:
        num_classes = len(indices)
    f, ax = plt.subplots(nrows=num_classes,
                         ncols=3,
                         sharex=True, sharey=True,
                         figsize=(6, 10),
                         dpi=DPI)
    plot_index = 0
    for class_index in range(len(class_labels)):
        if indices is not None and class_index not in indices:
            continue

        if plot_index == 0:
            # Add titles to top of plots
            ax[plot_index][0].set_title("Binary", fontsize=16)
            ax[plot_index][1].set_title("OVA", fontsize=16)
            ax[plot_index][2].set_title("Multi", fontsize=16)

        class_name = class_labels[class_index]
        plot_model_rates(class_name, binary_model, ax[plot_index][0])
        plot_model_rates(class_name, ova_model, ax[plot_index][1])
        plot_model_rates(class_name, multi_model, ax[plot_index][2])

        pretty_class_name = clean_class_name(class_name)
        ax[plot_index][0].text(-0.45, 0.81, pretty_class_name, fontsize=14)
        plot_index += 1

    y_indices = [0.1, 0.3, 0.5, 0.7, 0.9]
    y_ticks = ["10", "30", "50", "70", "90"]
    # x and y indices/ticks are the same
    plt.xticks(np.arange(5), y_ticks)
    plt.yticks(y_indices, y_ticks)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    mpl.rcParams['font.serif'] = ['times', 'times new roman']
    mpl.rcParams['font.family'] = 'serif'

    f.text(0.5, 0.08, 'Assigned Probability ' + r' $\pm10\%$', fontsize=14, ha='center')
    f.text(0.03, .5, r'Empirical Probability $\equiv$ TP/Total ($\%$)',
           fontsize=14, va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0, hspace=0)
    f.savefig(ROOT_DIR + "/output/custom_figures/merged_metrics" +
              str(indices) + ".pdf", bbox_inches='tight')
    plt.show()


def get_pc_per_range(model, class_name):
    """
    Get the purity and completeness for each 
    """
    class_total = model.class_counts[class_name]
    if model.num_runs is not None:
        class_total = model.num_runs * class_total * .33

    true_positives, totals = model.range_metrics_10[class_name]
    purities = []  # Accuracy per range (true positive/total)
    comps = []
    TP_count = 0
    total_count = 0

    for index in reversed(range(len(true_positives))):
        cur_p = 0  # Current purity
        cur_c = 0  # Current completeness
        TP_count += true_positives[index]
        total_count += totals[index]
        if total_count != 0:
            # positive class samples / totals # with prob in range
            cur_p = TP_count / total_count
        if class_total != 0:
            cur_c = TP_count / class_total

        purities.append(cur_p)
        comps.append(cur_c)
    purities.reverse()
    comps.reverse()
    return purities, comps


def plot_model_curves(class_name, model, range_metrics, ax):
    """
    Plots rates for this model/class on axis, with annotations 
    """
    def plot_axis(ax, data, color):
        """
        Plot data on axis in certain color
        """
        x_indices = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ax.scatter(x_indices, data, color=color, s=4)
        ax.plot(x_indices, data, color=color, linewidth=2)
        ax.set_yticks([])  # same for y ticks
        ax.set_ylim([0, 1])
    # Get balanced purities
    preds = np.concatenate(model.results)
    if model.name == "Binary Classifiers":
        purities = get_binary_balanced_purity_ranges(
            preds, model.class_labels, 0.1)[class_name]
    else:
        purities = get_balanced_purity_ranges(preds, model.class_labels, 0.1)[class_name]

    # Get completenesses
    comps = get_completeness_ranges(model, range_metrics, class_name)

    print("\n\n P-C metrics for : " + class_name)
    plot_axis(ax, comps, C_BAR_COLOR)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim([0, 1])
    plot_axis(ax2, purities, P_BAR_COLOR)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax2


def plot_pc_curves_together(binary_model, ova_model, multi_model, indices):
    """
    Plot balanced purity and completeness vs probability threshold for each class separately; combine this for all three classifiers onto a single figure.
    :param indices: class indices to plot (from class_labels)
    """
    raise ValueError(
        "Need to finish! Need to implement ranged balance purity for binary classifier. And test all ranged balanced purity functionality.")
    binary_range_metrics = binary_model.compute_probability_range_metrics(
        binary_model.results)
    ova_range_metrics = ova_model.compute_probability_range_metrics(
        ova_model.results)
    multi_range_metrics = multi_model.compute_probability_range_metrics(
        multi_model.results)

    class_labels = ova_model.class_labels
    f, ax = plt.subplots(nrows=len(indices),
                         ncols=3,
                         sharex=True, sharey=True,
                         figsize=(FIG_WIDTH, 10),
                         dpi=DPI)

    y_indices = [0, 0.2, 0.4, 0.6, 0.8, 1]
    y_ticks = ["0", "20", "40", "60", "80", ""]
    plot_index = 0
    for class_index, class_name in enumerate(class_labels):
        if class_index not in indices:
            continue

        if plot_index == 0:
            # Add titles to top of plots
            ax[plot_index][0].set_title("ova", fontsize=TICK_S)
            ax[plot_index][1].set_title("OVA", fontsize=TICK_S)
            ax[plot_index][2].set_title("Multi", fontsize=TICK_S)

        plot_model_curves(class_name, ova_model, ova_range_metrics,  ax[plot_index][0])
        plot_model_curves(class_name, ova_model, ova_range_metrics,  ax[plot_index][1])
        mirror_ax = plot_model_curves(
            class_name, multi_model, multi_range_metrics, ax[plot_index][2])

        ax[plot_index][0].set_yticks(ticks=y_indices)
        ax[plot_index][0].set_yticklabels(labels=y_ticks, color=P_BAR_COLOR)
        mirror_ax.set_yticks(ticks=y_indices)
        mirror_ax.set_yticklabels(labels=y_ticks, color=C_BAR_COLOR)
        ax[plot_index][0].tick_params(axis='both', direction='in', labelsize=10)
        ax[plot_index][1].tick_params(axis='both', direction='in')
        ax[plot_index][2].tick_params(axis='both', direction='in', labelsize=10)

        mpl.rcParams['font.serif'] = ['times', 'times new roman']
        mpl.rcParams['font.family'] = 'serif'
        pretty_class_name = clean_class_name(class_name)
        ax[plot_index][0].text(0, 0.85, pretty_class_name, fontsize=14)
        plot_index += 1

    x_indices = np.linspace(0, 1, 11)[:-1]

    plt.xticks(x_indices, ["", "10", "", "30", "", "50", "", "70", "", "90"])
    rc('text', usetex=True)
    f.text(0.5, 0.08, r'Probability $\geq$X\%', fontsize=TICK_S, ha='center')
    f.text(0.03, .5, 'Purity (\%)',
           fontsize=TICK_S, va='center', rotation='vertical', color=P_BAR_COLOR)
    f.text(0.98, .5, 'Completeness (\%)',
           fontsize=TICK_S, va='center', rotation='vertical', color=C_BAR_COLOR)

    plt.subplots_adjust(wspace=0, hspace=0)

    f.savefig("../output/custom_figures/merged_pc_curves_" +
              str(indices) + ".pdf", bbox_inches='tight')
    plt.show()

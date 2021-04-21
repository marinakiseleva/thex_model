"""
Helper function for computing different performance metrics
"""
import numpy as np
from sklearn.metrics import confusion_matrix

import utilities.utilities as thex_utils
from thex_data.data_consts import UNDEF_CLASS, ORDERED_CLASSES

# Computing helpers


def get_ordered_metrics(class_metrics, baselines=None, intervals=None):
    """
    Reorder metrics and reformat class names in hierarchy groupings
    :param class_metrics: Mapping from class name to metric value.
    :[optional] param baselines: Mapping from class name to random-baseline performance
    :[optional] param intervals: Mapping from class name to confidence intervals
    """
    ordered_formatted_names = []
    ordered_metrics = []
    ordered_baselines = [] if baselines is not None else None
    ordered_intervals = [] if intervals is not None else None
    for class_name in ORDERED_CLASSES:
        for c in class_metrics.keys():
            if c.replace(UNDEF_CLASS, "") == class_name:
                # Add to metrics and baselines
                ordered_metrics.append(class_metrics[c])
                if baselines is not None:
                    ordered_baselines.append(baselines[c])
                if intervals is not None:
                    ordered_intervals.append(intervals[c])

                pretty_class_name = clean_class_name(c)
                ordered_formatted_names.append(pretty_class_name)
                break

    ordered_formatted_names.reverse()
    ordered_metrics.reverse()
    if baselines is not None:
        ordered_baselines.reverse()
    if intervals is not None:
        ordered_intervals.reverse()
    return [ordered_formatted_names, ordered_metrics, ordered_baselines, ordered_intervals]


def compute_baselines(class_counts, class_labels, N, balanced_purity, class_priors=None):
    """
    Get random classifier baselines for completeness and purity
    Completeness: 1/ number of classes
    Purity: class count/ total count
    Balanced Purity: 1/ number of classes
    :param N: Number of classes
    """
    comp_baselines = {}
    purity_baselines = {}
    total_count = sum(class_counts.values())
    for class_name in class_labels:
        if class_priors is not None:
            # If class priors are used - use relative frequency of classes
            class_rate = class_counts[class_name] / total_count
        else:
            class_rate = 1 / N

        # Compute baselines
        comp_baselines[class_name] = class_rate
        if balanced_purity:
            purity_baselines[class_name] = 1 / N
        else:
            # Random purity baseline is class count/ total
            purity_baselines[class_name] = class_counts[class_name] / total_count

    return comp_baselines, purity_baselines


def get_accuracy(class_metrics, N):
    """
    Get accuracy as total # of TP / total # of samples
    :param class_metrics : Map from class name to metrics (which are map from TP, TN, FP, FN to values)
    :param N: total number of samples
    """
    if N == 0:
        return 0
    TP_total = 0
    for class_name in class_metrics.keys():
        metrics = class_metrics[class_name]
        TP = metrics["TP"]
        TP_total += TP
    return TP_total / N


def get_class_name(labels, class_names):
    """
    get class name for set of labels
    """
    for class_name in class_names:
        if class_name in thex_utils.convert_str_to_list(labels):
            return class_name


def compute_balanced_purity(preds, class_labels):
    """
    Get completeness & balanced purity for each class, return as 2 dicts.
    :param preds: List of Numpy rows of probabilites (last col is label) 
    :param class_metrics: Map from class name to metrics (which are map from TP, TN, FP, FN to values)
    """
    purities = {}
    # assignments: get number of each class predicted as key class
    # key: predicted class (so all samples in the set are predicted as the
    # key), value: map from class name to # of samples predicted as key
    assignments = {cn: {p: 0 for p in class_labels} for cn in class_labels}
    label_index = len(class_labels)
    class_counts = {cn: 0 for cn in class_labels}
    for row in preds:
        labels = row[label_index]
        max_class_index = np.argmax(row[:len(row) - 1])
        max_class_name = class_labels[max_class_index]
        true_class = get_class_name(labels, class_labels)
        assignments[max_class_name][true_class] += 1
        class_counts[true_class] += 1

    for class_name in class_labels:
        # All assignments for this class.
        class_assgs = assignments[class_name]
        TPR = class_assgs[class_name] / class_counts[class_name]
        den = 0
        for ck in class_assgs.keys():
            den += class_assgs[ck] / class_counts[ck]
        purities[class_name] = TPR / den
    return purities


def compute_performance(class_metrics):
    """
    Get completeness & purity for each class, return as 2 dicts.
    Completeness = TP/(TP+FN) 
    Purity = TP/(TP+FP)
    :param class_metrics: Map from class name to metrics (which are map from TP, TN, FP, FN to values)
    """
    purities = {}
    comps = {}
    for class_name in class_metrics.keys():
        metrics = class_metrics[class_name]
        TP = metrics["TP"]
        FP = metrics["FP"]
        FN = metrics["FN"]
        # Ensure there are some samples of this class
        if TP + FN == 0:
            raise ValueError("No samples for class " + class_name)
        comps[class_name] = TP / (TP + FN)
        purities[class_name] = TP / (TP + FP) if TP + FP > 0 else 0
    return purities, comps


def compute_confusion_matrix(results, class_labels):
    """
    Compute confusion matrix using sklearn defined function
    :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of class_labels & the last column containing the full, true label
    """
    results = np.concatenate(results)

    label_index = len(class_labels)  # Last column is label

    predictions = []  # predictions (as class indices)
    labels = []  # labels (as class indices)
    for row in results:
        row_labels = thex_utils.convert_str_to_list(row[label_index])
        # Sample is an instance of this current class.
        true_labels = list(set(class_labels).intersection(set(row_labels)))
        if len(true_labels) != 1:
            raise ValueError("Class has more than 1 label.")
        true_label = true_labels[0]

        # Get class index of max prob; exclude last column since it is label
        pred_class_index = np.argmax(row[: len(row) - 1])
        actual_class_index = class_labels.index(true_label)

        # Use class index as label
        predictions.append(pred_class_index)
        labels.append(actual_class_index)

    index_labels = list(range(len(class_labels)))
    cm = confusion_matrix(labels,
                          predictions,
                          labels=index_labels,
                          normalize='true')
    return cm


def compute_confintvls(all_pc, class_labels):
    """
    Calculate 1sigma error bars for each class
    [µ − σ, µ + σ], where sigma is the standard deviation

    :param all_pc: List with length of N, each item is [pmap, cmap] for that fold/trial, where pmap is map from class name to purity
    """

    def get_cis(values, N):
        """
        Calculate confidence intervals [µ − 1.96*SEM, µ + 1.96*SEM] where
        SEM = σ/sqrt(N) 
        σ = sqrt( (1/ N ) ∑_n (a_i − µ)^2 )
        :param N: number of runs or folds.
        """
        mean = sum(values) / len(values)
        stdev = np.sqrt((sum((np.array(values) - mean) ** 2)) / (N - 1))
        SEM = stdev / np.sqrt(N)
        # 95% confidence intervals, [µ − 1.96σ, µ + 1.96σ]
        low_CI = mean - (1.96 * SEM)
        high_CI = mean + (1.96 * SEM)
        if low_CI < 0:
            low_CI = 0
        if high_CI > 1:
            high_CI = 1
        return [low_CI, high_CI]

    N = len(all_pc)  # Number of folds/trials

    purity_cis = {}
    comp_cis = {}

    for class_name in class_labels:
        N_p = N
        class_purities = []
        class_comps = []
        for pc in all_pc:
            class_purity = pc[0][class_name]
            class_compeleteness = pc[1][class_name]
            if class_purity is not None:
                class_purities.append(class_purity)
            else:
                print("No measurable purity for " + class_name)
                N_p = N_p - 1

            if class_compeleteness is None:
                raise ValueError("Completeness should never be None for " + class_name)

            class_comps.append(class_compeleteness)
        # Calculate confidence intervals
        purity_cis[class_name] = get_cis(class_purities, N_p)
        comp_cis[class_name] = get_cis(class_comps, N)

    thex_utils.pretty_print_dict(purity_cis, "Purity confidence intervals")
    thex_utils.pretty_print_dict(comp_cis, "Completeness confidence intervals")
    return purity_cis, comp_cis


def get_agg_prob_vs_class_rates(total_count_per_range,  class_labels, class_positives, class_prob_rates, weighted):
    """
    Get aggregated probability vs class rates
    """
    length = len(class_positives[class_labels[0]])
    # Plot aggregated prob vs rates across all classes using weighted averages
    aggregated_rates = np.zeros(length)
    for class_name in class_labels:
        class_weights = np.array(class_positives[
            class_name]) / total_count_per_range
        pos_prob_rates = np.array(class_prob_rates[class_name])
        if weighted:
            # Weighted rate = weight * rates
            aggregated_rates += np.multiply(class_weights, pos_prob_rates)
        else:
            # Balanced average = (1/K) * rates
            aggregated_rates += (1 / len(class_labels)) * pos_prob_rates
    return aggregated_rates

# Plotting helpers


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


# Density analysis helpers

def clean_plot(y, ax, name, color):
    """
    Helper plotting function for density analysis
    """
    def pre_plot_clean(x, y):
        """
        Keep only x, y values where y is not None
        """
        new_x = []
        new_y = []
        for index, value in enumerate(x):
            if y[index] is not None:
                new_x.append(value)
                new_y.append(y[index])
        return new_x, new_y

    orig_x = list(range(0, 100, 1))
    x, y = pre_plot_clean(orig_x,  y)
    print("\nPlotting " + str(name) + " versus % top densities. Y values:")
    print(y)
    ax.scatter(x, y, color=color, s=2)
    ax.plot(x, y, color=color, label=name)


def get_proportion_results(class_labels, indices, results):
    """
    Reduce results (as probabilities) to those with these indices 
    :param indices: List of indices to keep 
    :param results: Results, with density per class and last column has label 
    """
    probs_only = results[:, 0:len(class_labels)].astype(float)

    # Select rows at those indices
    densities = np.take(probs_only, indices=indices, axis=0)

    # Normalize these densities to get probabilities
    probs = densities / densities.sum(axis=1)[:, None]

    labels = np.take(results,
                     indices=indices,
                     axis=0)[:, len(class_labels)]

    # Put probs & labels in same Numpy array
    prop_results = np.hstack((probs, labels.reshape(-1, 1)))
    return prop_results


def get_average(metrics):
    """
    Gets average if there are values, otherwise 0
    """
    avg = 0
    valid_count = 0
    for class_name in metrics.keys():
        if metrics[class_name] is not None:
            avg += metrics[class_name]
            valid_count += 1
    if valid_count > 0:
        return avg / valid_count
    else:
        return None

"""
Helper function for computing different performance metrics
"""

from sklearn.metrics import confusion_matrix
import numpy as np

import utilities.utilities as thex_utils
from thex_data.data_consts import UNDEF_CLASS, ORDERED_CLASSES


def update_class_purities(class_purities, class_metrics, i):
    """
    Add next purity to list of purities for each class. If not valid, append None. If there are no new samples (no change in TP+FP), also append None.
    :param class_purities: Map from class name to list, where we will append [TP, TP+FP, i, class count] based on class metrics
    :param class_metrics: Map from class name to metrics (which are map from TP, TN, FP, FN to values)
    """
    for class_name in class_purities.keys():
        metrics = class_metrics[class_name]
        den = metrics["TP"] + metrics["FP"]
        v = None
        if den > 0:
            # Get last value, and make sure (TP+FP) counts have changed
            last_val = class_purities[class_name][-1]
            if last_val is None or last_val[1] < den:
                class_count = metrics["TP"] + metrics["FN"]
                v = [metrics["TP"], den, i, class_count]

        class_purities[class_name].append(v)
    return class_purities


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


def compute_performance(class_metrics):
    """
    Get recall (completeness), precision (purity), specificity (true negative rate)   - for only the classes included in the sample. If no samples, return None for that metric. 
    :param class_metrics: Map from class name to metrics (which are map from TP, TN, FP, FN to values) 
    """
    precisions = {}
    recalls = {}
    for class_name in class_metrics.keys():
        metrics = class_metrics[class_name]
        TP = metrics["TP"]
        FP = metrics["FP"]
        TN = metrics["TN"]
        FN = metrics["FN"]

        # Only if there are samples in this class, calculate its metrics
        total = TP + TN + FP + FN
        recalls[class_name] = TP / (TP + FN) if (TP + FN) > 0 else None
        precisions[class_name] = TP / (TP + FP) if (TP + FP) > 0 else None
    return recalls, precisions


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


def compute_confintvls(set_totals, N, class_labels):
    """
    Compute 95% confidence intervals, [µ − 2σ, µ + 2σ],
    for each class
    :param set_totals: Map from fold # to map of metrics
    :param N: Number of runs or folds
    """
    if N is None:
        return None, None

    def get_cis(values, N):
        """
        Calculate confidence intervals [µ − 2σ, µ + 2σ] where 
        σ = sqrt( (1/ N ) ∑_n (a_i − µ)^2 )
        :param N: number of runs
        """
        mean = sum(values) / len(values)
        a = sum((np.array(values) - mean) ** 2)
        stdev = np.sqrt((a / (N - 1)))
        stderr = stdev / np.sqrt(N)
        # 95% confidence intervals, [µ − 1.96σ, µ + 1.96σ]
        return [mean - (1.96 * stderr), mean + (1.96 * stderr)]

    prec_cis = {cn: [0, 0] for cn in class_labels}
    recall_cis = {cn: [0, 0] for cn in class_labels}
    for class_name in set_totals.keys():
        precisions = []
        recalls = []
        for fold_num in set_totals[class_name].keys():
            metrics = set_totals[class_name][fold_num]
            den = metrics["TP"] + metrics["FP"]
            prec = metrics["TP"] / den if den > 0 else 0
            precisions.append(prec)
            den = metrics["TP"] + metrics["FN"]
            rec = metrics["TP"] / den if den > 0 else 0
            recalls.append(rec)

        # Calculate confidence intervals
        prec_cis[class_name] = get_cis(precisions, N)
        recall_cis[class_name] = get_cis(recalls, N)
    return prec_cis, recall_cis


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

                pretty_class_name = c
                if UNDEF_CLASS in c:
                    pretty_class_name = class_name.replace(UNDEF_CLASS, "")
                    pretty_class_name = pretty_class_name + " (unspecified)"
                ordered_formatted_names.append(pretty_class_name)
                break

    ordered_formatted_names.reverse()
    ordered_metrics.reverse()
    if baselines is not None:
        ordered_baselines.reverse()
    if intervals is not None:
        ordered_intervals.reverse()
    return [ordered_formatted_names, ordered_metrics, ordered_baselines, ordered_intervals]

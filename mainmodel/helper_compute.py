"""
Helper function for computing different performance metrics
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
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


def get_completeness_ranges(model, range_metrics, class_name):
    """
    Get completeness, for each probability range threshold, for class class_name
    """
    true_positives, totals = range_metrics[class_name]
    class_total = model.class_counts[class_name]
    comps = []
    TP_count = 0
    total_count = 0
    for index in reversed(range(len(true_positives))):
        cur_c = 0  # Current completeness
        TP_count += true_positives[index]
        total_count += totals[index]
        if total_count != 0:
            # positive class samples / totals # with prob in range
            cur_p = TP_count / total_count
        if class_total != 0:
            cur_c = TP_count / class_total
        comps.append(cur_c)
    comps.reverse()
    return comps


def get_puritys_and_comps(class_metrics):
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


#############################################################
#### BALANCED PURITY HELPERS ########################


def compute_binary_balanced_purity(preds, class_labels):
    """
    Get completeness & balanced purity for each class, return as 2 dicts.
    balanced purity = TPR/ (TPR+FPR)
    :param preds: List of Numpy rows of probabilites (last col is label)  
    """
    purities = {}
    # assignments: get number of class predicted as key class vs. number of
    # NOT class predicted as key class
    assignments = {cn: {cn: 0, "NOT": 0} for cn in class_labels}
    label_index = len(class_labels)
    class_counts = {cn: 0 for cn in class_labels}
    for row in preds:
        labels = row[label_index]
        max_class_index = np.argmax(row[:len(row) - 1])
        max_class_name = class_labels[max_class_index]
        true_class = thex_utils.get_class_name(labels, class_labels)
        if max_class_name == true_class:
            assignments[max_class_name][true_class] += 1
        else:
            assignments[max_class_name]["NOT"] += 1
        class_counts[true_class] += 1

    total = sum(class_counts.values())
    for class_name in class_labels:
        # All assignments for this class.
        class_assgs = assignments[class_name]
        TPR = class_assgs[class_name] / class_counts[class_name]
        FPR = class_assgs["NOT"] / (total - class_counts[class_name])
        den = TPR + FPR
        # if den is 0, there is no purity measure because nothing was classified
        # as this class
        purities[class_name] = TPR / den if den > 0 else None
    return purities


def compute_balanced_purity(preds, class_labels, model_name):
    """
    Get balanced purity for each class, over all preds.
    :param preds: List of Numpy rows of probabilites (last col is label) 
    """
    if model_name == "Binary Classifiers":
        return compute_binary_balanced_purity(preds, class_labels)
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
        true_class = thex_utils.get_class_name(labels, class_labels)
        assignments[max_class_name][true_class] += 1
        class_counts[true_class] += 1

    for class_name in class_labels:
        # All assignments for this class.
        class_assgs = assignments[class_name]
        TPR = class_assgs[class_name] / class_counts[class_name]
        den = 0
        for ck in class_assgs.keys():
            den += class_assgs[ck] / class_counts[ck]
        # if den is 0, there is no purity measure because nothing was classified
        # as this class
        purities[class_name] = TPR / den if den > 0 else None
    return purities


############################################
# Balanced purity helpers for ranged metrics (P-C curves)

def bp_get_class_stats(target_class, prob_range, rows, class_labels):
    """
    Returns list in order of class_labels, where each value corresponds to number of samples of that class predicted as target class with a probability in the range prob_range.
    Last value of the list is total # of rows with true label of target_class
    :param prob_range: [min_prob_inclusive, max_prob_exclusive]
    :param rows: 2d array, each row is probabilities assigned in order of class_labels, and last column is true label
    """
    label_index = len(class_labels)
    # Get index of target_class in class_labels
    target_class_index = class_labels.index(target_class)

    class_counts = OrderedDict()
    for cn in class_labels:
        class_counts[cn] = 0
    target_class_count = 0
    # Iterate over all given rows (each row is probs assigned to classes, and
    # class label).
    for row in rows:
        true_class = thex_utils.get_class_name(row[label_index], class_labels)
        target_class_prob = row[target_class_index]

        min_prob_inc = prob_range[0]
        max_prob_exc = prob_range[1]

        max_class_index = np.argmax(row[:len(row) - 1])
        max_class_prob = row[max_class_index]
        # if the probability assigned to target_class is within the prob_range,
        if target_class_prob >= min_prob_inc and target_class_prob < max_prob_exc:
            # add 1 to that row's true class label's count.
            max_class_name = class_labels[max_class_index]
            # Count this class if the max prob is for the target class
            if max_class_name == target_class:
                class_counts[true_class] += 1

        # Find all samples with target_class label, and count those that are assigned a probability in this range
        # to any class
        if true_class == target_class and max_class_prob >= min_prob_inc and max_class_prob < max_prob_exc:
            target_class_count += 1

    return list(class_counts.values()) + [target_class_count]


def bp_get_assignments(preds, class_labels, bin_size):
    """
    Gets # of samples of each class assigned to each target class, by probability range.
    :param preds: List of Numpy rows of probabilites (last col is label) 
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, usually 0.1
    """
    assignments = []
    for i in range(int(1 / (bin_size))):
        # bins are left inclusive, bin is 0 <= x < .1. ; so last bin <=1.01
        # Need to round due to strange rounding issue where 0.6 becomes 0.6000...001.
        min_prob_inc = round(i * bin_size, 2)
        max_prob_exc = round(min_prob_inc + bin_size, 2)
        range_a = []
        for target_class in class_labels:
            ca = bp_get_class_stats(
                target_class, [min_prob_inc, max_prob_exc], preds, class_labels)
            range_a.append(ca)
        assignments.append(range_a)
    return assignments


def bp_aggregate_assignments(assignments, class_labels, num_bins):
    """
    Aggregate assignments from max prob to lower probs (to make it thresholded, <=)
    Return list in order of class_labels, where each value corresponds to number of samples of that class
    assigned a probability >= min of that range's prob_range for target_class AND predicted as target class.
    """
    agg_assignments = np.zeros(
        shape=(num_bins, len(class_labels), len(class_labels) + 1))

    for index in reversed(range(num_bins)):
        for target_class_index, target_class in enumerate(class_labels):
            agg_assignments[index][target_class_index] = assignments[
                index][target_class_index]
            # If there is an index after this, add those (this is the aggregation)
            if index < (num_bins - 1):
                agg_assignments[index][target_class_index] = np.add(
                    agg_assignments[index][target_class_index], agg_assignments[index + 1][target_class_index])
    return agg_assignments


def bp_get_range_bps(assignments, class_labels):
    """
    Get balanced purity for each probability range threshold, and each class. For class A, get balanced purity at probs >=10%, probs>=20% and so on. Return as list, in order of ranges; each list contains dict from target/predicted class name to balanced purity at threshold.
    :param assignments: aggregated assignments from bp_aggregate_assignments
    :param class_labels: labels in correct order.
    """
    range_class_bps = []
    class_count_index = len(class_labels)
    for prob_range_assgns in assignments:
        class_bps = {cn: [] for cn in class_labels}
        for targ_class_index, targ_class in enumerate(class_labels):
            targ_class_assgns = prob_range_assgns[targ_class_index]

            TPR = targ_class_assgns[targ_class_index] / \
                targ_class_assgns[class_count_index]
            FPR = 0
            for alt_class_index, alt_class in enumerate(class_labels):
                if alt_class == targ_class:
                    continue
                # of samples of alt_class predicted as targ_class divided by
                # total number of alt_class samples assigned targ_class prob in this
                # range as max prob
                alt_total = prob_range_assgns[alt_class_index][class_count_index]
                FPR += targ_class_assgns[alt_class_index] / alt_total
            class_bps[targ_class] = TPR / (TPR + FPR)
        range_class_bps.append(class_bps)
    return range_class_bps


def get_balanced_purity_ranges(preds, class_labels, bin_size):
    """
    Get balanced purity for each class, for each range of probabilities, for multiclass classifiers. 
    :param preds: List of Numpy rows of probabilites (last col is label) ; comes from np.concatenate(model.results)
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, usually 0.1
    """

    assignments = bp_get_assignments(preds, class_labels, bin_size)

    agg_assignments = bp_aggregate_assignments(assignments, class_labels, num_bins=10)

    range_class_bps = bp_get_range_bps(agg_assignments, class_labels)
    # Reformat, so that instead of list of dicts, it is a dict from class name to list.
    class_to_bps = {}
    for class_name in class_labels:
        cur_class_bps = []
        for cur_range_bp in range_class_bps:
            cur_class_bps.append(cur_range_bp[class_name])
        class_to_bps[class_name] = cur_class_bps
    return class_to_bps


def bp_binary_get_class_stats(target_class, prob_range, rows, class_labels):
    """
    Return list of length 4 with counts: [TP for target_class,  FP, P of target_class, N negatives of target_class]
    """
    label_index = len(class_labels)
    # Get index of target_class in class_labels
    target_class_index = class_labels.index(target_class)

    TP_count = 0
    FP_count = 0
    target_class_count = 0
    non_target_class_count = 0
    # Iterate over all given rows (each row is probs assigned to classes, and
    # class label).
    for row in rows:
        true_class = thex_utils.get_class_name(row[label_index], class_labels)
        target_class_prob = row[target_class_index]

        min_prob_inc = prob_range[0]
        max_prob_exc = prob_range[1]

        max_class_index = np.argmax(row[:len(row) - 1])
        max_class_prob = row[max_class_index]
        max_class_name = class_labels[max_class_index]
        # if the probability assigned to target_class is within the prob_range,
        # and it is the max prob
        if target_class_prob >= min_prob_inc and target_class_prob < max_prob_exc and max_class_name == target_class:
                # Add if TP or FP
            if true_class == target_class:
                TP_count += 1
            elif true_class != target_class:
                FP_count += 1

        # is target_class and assigned a probability in this range
        # to any class
        if true_class == target_class and max_class_prob >= min_prob_inc and max_class_prob < max_prob_exc:
            target_class_count += 1
        elif true_class != target_class and max_class_prob >= min_prob_inc and max_class_prob < max_prob_exc:
            non_target_class_count += 1

    return [TP_count, FP_count, target_class_count, non_target_class_count]


def bp_binary_get_assignments(preds, class_labels, bin_size):
    """
    Gets [TP, FP, total P, total N] for each class, in order of class_labels.
    :param preds: List of Numpy rows of probabilites (last col is label) 
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, usually 0.1
    """
    assignments = []
    for i in range(int(1 / (bin_size))):
        # bins are left inclusive, bin is 0 <= x < .1. ; so last bin <=1.01
        # Need to round due to strange rounding issue where 0.6 becomes 0.6000...001.
        min_prob_inc = round(i * bin_size, 2)
        max_prob_exc = round(min_prob_inc + bin_size, 2)
        range_a = []
        for target_class in class_labels:
            ca = bp_binary_get_class_stats(
                target_class, [min_prob_inc, max_prob_exc], preds, class_labels)
            range_a.append(ca)
        assignments.append(range_a)
    return assignments


def bp_binary_aggregate_assignments(assignments, class_labels, num_bins):
    """
    Aggregate assignments from max prob to lower probs (to make it thresholded, <=)
    :param assignments: [TP, FP, total P, total N] for each class, in order of class_labels.
    """
    agg_assignments = np.zeros(
        shape=(num_bins, len(class_labels), 4))

    for index in reversed(range(num_bins)):
        for target_class_index, target_class in enumerate(class_labels):
            agg_assignments[index][target_class_index] = assignments[
                index][target_class_index]
            # If there is an index after this, add those (this is the aggregation)
            if index < (num_bins - 1):
                agg_assignments[index][target_class_index] = np.add(
                    agg_assignments[index][target_class_index], agg_assignments[index + 1][target_class_index])
    return agg_assignments


def bp_binary_get_range_bps(assignments, class_labels):
    """
    Get balanced purity for each probability range threshold, and each class. For class A, get balanced purity at probs >=10%, probs>=20% and so on. Return as list, in order of ranges; each list contains dict from target/predicted class name to balanced purity at threshold.
    :param assignments: [TP, FP, total P, total N] for each class, in order of class_labels from bp_binary_get_assignments
    :param class_labels: labels in correct order.
    """
    range_class_bps = []
    class_count_index = len(class_labels)
    for prob_range_assgns in assignments:
        class_bps = {cn: [] for cn in class_labels}
        for targ_class_index, targ_class in enumerate(class_labels):
            targ_class_assgns = prob_range_assgns[targ_class_index]

            TP = targ_class_assgns[0]
            FP = targ_class_assgns[1]
            P = targ_class_assgns[2]
            N = targ_class_assgns[3]

            TPR = TP / P
            FPR = FP / N
            class_bps[targ_class] = TPR / (TPR + FPR)

        range_class_bps.append(class_bps)
    return range_class_bps


def get_binary_balanced_purity_ranges(preds, class_labels, bin_size):
    """
    Get balanced purity for each class, for each range of probabilities, for binary classifier. 
    Return dict. of class names to balanced purity at each prob threshold (10 thresholds)
    :param preds: List of Numpy rows of probabilites (last col is label) 
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, usually 0.1
    """
    assignments = bp_binary_get_assignments(preds, class_labels, bin_size)

    agg_assignments = bp_binary_aggregate_assignments(
        assignments, class_labels, num_bins=10)

    range_class_bps = bp_binary_get_range_bps(agg_assignments, class_labels)
    # Reformat, so that instead of list of dicts, it is a dict from class name to list.
    class_to_bps = {}
    for class_name in class_labels:
        cur_class_bps = []
        for cur_range_bp in range_class_bps:
            cur_class_bps.append(cur_range_bp[class_name])
        class_to_bps[class_name] = cur_class_bps
    return class_to_bps
##########################################################################


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

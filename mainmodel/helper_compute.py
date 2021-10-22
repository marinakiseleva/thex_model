"""
Helper function for computing different performance metrics
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import utilities.utilities as thex_utils
from thex_data.data_consts import UNDEF_CLASS, ORDERED_CLASSES

# Computing helpers


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


def get_completeness_ranges(class_counts, range_metrics, class_name):
    """
    Get completeness, for each probability range threshold, for class class_name
    :param range_metrics: dict from class name : TPs and Total Ps for each prob range 
    """
    true_positives, totals = range_metrics[class_name]
    class_total = class_counts[class_name]
    comps = []
    TP_count = 0
    for index in reversed(range(len(true_positives))):
        cur_c = 0  # Current completeness
        TP_count += true_positives[index]
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
    :param class_metrics: Map from class name to metrics (which is map from TP, TN, FP, FN to values)
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
            print("No samples for class " + class_name)
        comps[class_name] = TP / (TP + FN) if TP + FN > 0 else 0
        purities[class_name] = TP / (TP + FP) if TP + FP > 0 else 0
    return purities, comps


#############################################################
#### BALANCED PURITY HELPERS ########################


def compute_binary_balanced_purity(preds, class_labels):
    """
    Get balanced purity for each class, dict from class name to balanced purity.
    balanced purity = TPR/ (TPR+FPR)
    :param preds: List of Numpy rows of probabilites (last col is label)  
    """

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
    purities = OrderedDict()
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
            den += class_assgs[ck] / class_counts[ck] if class_counts[ck] > 0 else 0
        # if den is 0, there is no purity measure because nothing was classified
        # as this class
        purities[class_name] = TPR / den if den > 0 else None
    return purities


############################################
# Balanced purity helpers for ranged metrics (P-C curves)

def bp_get_class_stats(target_class, prob_range, rows, class_labels, EP=False):
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
            # If we are doing empirical probs, we always count it.
            if EP or max_class_name == target_class:
                class_counts[true_class] += 1

    return list(class_counts.values())


def bp_get_assignments(preds, class_labels, bin_size, EP=False):
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
                target_class, [min_prob_inc, max_prob_exc], preds, class_labels, EP)
            range_a.append(ca)
        assignments.append(range_a)
    return assignments


def bp_aggregate_assignments(assignments, class_labels, num_bins):
    """
    Aggregate assignments from max prob to lower probs (to make it thresholded, <=)
    Returns a list of lists where eahc list is for a target_class in class_labels and contains:
        for each class in class_labels, the number of samples of that class that were assigned a probability in range for the target_class. 

    """
    agg_assignments = np.zeros(
        shape=(num_bins, len(class_labels), len(class_labels)))

    for index in reversed(range(num_bins)):
        for target_class_index, target_class in enumerate(class_labels):
            agg_assignments[index][target_class_index] = assignments[
                index][target_class_index]
            # If there is an index after this, add those (this is the aggregation)
            if index < (num_bins - 1):
                agg_assignments[index][target_class_index] = np.add(
                    agg_assignments[index][target_class_index], agg_assignments[index + 1][target_class_index])

    return agg_assignments


def bp_get_range_bps(assignments, class_labels, total_class_counts):
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
                total_class_counts[targ_class]
            FPR = 0
            for alt_class_index, alt_class in enumerate(class_labels):
                if alt_class == targ_class:
                    continue
                alt_cn = class_labels[alt_class_index]
                FPR += targ_class_assgns[alt_class_index] / total_class_counts[alt_cn]

            class_bps[targ_class] = TPR / (TPR + FPR) if TPR + FPR > 0 else 0

        range_class_bps.append(class_bps)
    return range_class_bps


def get_balanced_purity_ranges(preds, class_labels, bin_size, total_class_counts):
    """
    Get balanced purity for each class, for each range of probabilities, for multiclass classifiers. 
    :param preds: List of Numpy rows of probabilites (last col is label) ; comes from np.concatenate(model.results)
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, usually 0.1
    """

    assignments = bp_get_assignments(preds, class_labels, bin_size)

    agg_assignments = bp_aggregate_assignments(assignments, class_labels, num_bins=10)

    range_class_bps = bp_get_range_bps(agg_assignments, class_labels, total_class_counts)
    # Reformat, so that instead of list of dicts, it is a dict from class name to list.
    class_to_bps = {}
    for class_name in class_labels:
        cur_class_bps = []
        for cur_range_bp in range_class_bps:
            cur_class_bps.append(cur_range_bp[class_name])
        class_to_bps[class_name] = cur_class_bps
    return class_to_bps


def get_multi_emp_prob_rates(preds, class_labels, bin_size, total_class_counts):
    """
    Get balanced purity / empirical probability for each class, for each range of probabilities, for multiclass classifiers. Unlike get_balanced_purity_ranges, this keeps it in respective bins for plotting. 
    :param preds: List of Numpy rows of probabilites (last col is label) ; comes from np.concatenate(model.results)
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, =0.2
    """

    assignments = bp_get_assignments(preds, class_labels, bin_size, True)
    range_class_bps = bp_get_range_bps(assignments, class_labels, total_class_counts)
    # Reformat, so that instead of list of dicts, it is a dict from class name to list.
    class_to_bps = {}
    for class_name in class_labels:
        cur_class_bps = []
        for cur_range_bp in range_class_bps:
            cur_class_bps.append(cur_range_bp[class_name])
        class_to_bps[class_name] = cur_class_bps
    return class_to_bps


def bp_binary_get_class_stats(target_class, prob_range, rows, class_labels, EP=False):
    """
    Return list of [P_r, N_r]
    Return list of 2 values:
    Rows with label=target_class assigned probability to target class in this range (P_r)
    Rows with label!=target_class assigned probability to target class in this range (N_r)
    """
    label_index = len(class_labels)
    # Get index of target_class in class_labels
    target_class_index = class_labels.index(target_class)
    min_prob_inc = prob_range[0]
    max_prob_exc = prob_range[1]

    P_r = 0
    N_r = 0
    for row in rows:
        true_class = thex_utils.get_class_name(row[label_index], class_labels)
        target_class_prob = row[target_class_index]

        if target_class_prob >= min_prob_inc and target_class_prob < max_prob_exc:
            if EP:
                if true_class == target_class:
                    P_r += 1
                else:
                    N_r += 1
            else:
                # For bal. purity: Only count as TP if target_class_prob was the max prob
                max_class_index = np.argmax(row[:len(row) - 1])
                pred_class_name = class_labels[max_class_index]
                if pred_class_name == target_class:
                    if true_class == target_class:
                        P_r += 1
                    else:
                        N_r += 1

    return [P_r, N_r]


def bp_binary_get_assignments(preds, class_labels, bin_size, EP=False):
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
                target_class, [min_prob_inc, max_prob_exc], preds, class_labels, EP)
            range_a.append(ca)
        assignments.append(range_a)
    return assignments


def bp_binary_aggregate_assignments(assignments, class_labels, num_bins):
    """
    Aggregate assignments from max prob to lower probs (to make it thresholded, <=)
    :param assignments: [P_r or TP_r, N_r] for each class, in order of class_labels.
    """
    agg_assignments = np.zeros(
        shape=(num_bins, len(class_labels), 2))

    for index in reversed(range(num_bins)):
        for target_class_index, target_class in enumerate(class_labels):
            agg_assignments[index][target_class_index] = assignments[
                index][target_class_index]
            # If there is an index after this, add those (this is the aggregation)
            if index < (num_bins - 1):
                agg_assignments[index][target_class_index] = np.add(
                    agg_assignments[index][target_class_index], agg_assignments[index + 1][target_class_index])
    return agg_assignments


def bp_binary_get_range_bps(assignments, class_labels, total_class_counts):
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
            TP = prob_range_assgns[targ_class_index][0]
            FP = prob_range_assgns[targ_class_index][1]

            P = total_class_counts[targ_class]
            N = sum(total_class_counts.values()) - P

            TPR = TP / P
            FPR = FP / N

            class_bps[targ_class] = TPR / (TPR + FPR) if TPR + FPR > 0 else 0

        range_class_bps.append(class_bps)
    return range_class_bps


def get_binary_balanced_purity_ranges(preds, class_labels, bin_size, total_class_counts):
    """
    Get balanced purity for each class, for each range of probabilities, for binary classifier. 
    Return dict. of class names to balanced purity at each prob threshold (10 thresholds)
    :param preds: List of Numpy rows of probabilites (last col is label) ; can be derived from model using preds = np.concatenate(model.results)
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, usually 0.1
    """
    assignments = bp_binary_get_assignments(preds, class_labels, bin_size)

    agg_assignments = bp_binary_aggregate_assignments(
        assignments, class_labels, num_bins=10)

    range_class_bps = bp_binary_get_range_bps(
        agg_assignments, class_labels, total_class_counts)
    # Reformat, so that instead of list of dicts, it is a dict from class name to list.
    class_to_bps = {}
    for class_name in class_labels:
        cur_class_bps = []
        for cur_range_bp in range_class_bps:
            cur_class_bps.append(cur_range_bp[class_name])
        class_to_bps[class_name] = cur_class_bps
    return class_to_bps


def get_binary_emp_prob_rates(preds, class_labels, bin_size, total_class_counts):
    """
    Get balanced purity / empirical probability for each class, for each range of probabilities, for binary classifiers. Unlike get_binary_balanced_purity_ranges, this keeps it in respective bins for plotting. 
    :param preds: List of Numpy rows of probabilites (last col is label) ; comes from np.concatenate(model.results)
    :param class_labels: class names in order to be treated
    :param bin_size: width of probability bin, =0.2
    """

    assignments = bp_binary_get_assignments(preds, class_labels, bin_size, True)
    range_class_bps = bp_binary_get_range_bps(
        assignments, class_labels, total_class_counts)
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


def compute_confintvls(all_pc, class_labels, balanced_purity):
    """
    Calculate 2*sigma error bars for each class
    [µ − σ, µ + σ], where sigma is the standard deviation
    :param all_pc: List with length of N, each item is [bpmap, pmap, cmap] for that fold/trial (balanced purity, purity, completeness maps)
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
            if balanced_purity:
                class_purity = pc[0][class_name]
            else:
                class_purity = pc[1][class_name]

            if class_purity is not None:
                class_purities.append(class_purity)
            else:
                print("No measurable purity for " + class_name)
                N_p = N_p - 1

            class_compeleteness = pc[2][class_name]
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

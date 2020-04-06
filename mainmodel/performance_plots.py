
"""
Class Mixin for MainModel which contains all performance plotting functionality
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import utilities.utilities as thex_utils


from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, TREE_ROOT, UNDEF_CLASS, ORDERED_CLASSES


class MainModelVisualization:
    """
    Mixin Class for MainModel performance visualization
    """

    def compute_confusion_matrix(self, results):
        """
        Compute confusion matrix using sklearn defined function
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """
        results = np.concatenate(results)

        label_index = len(self.class_labels)  # Last column is label

        predictions = []  # predictions (as class indices)
        labels = []  # labels (as class indices)
        for row in results:
            row_labels = thex_utils.convert_str_to_list(row[label_index])

            # Sample is an instance of this current class.
            true_labels = list(set(self.class_labels).intersection(set(row_labels)))
            if len(true_labels) != 1:
                raise ValueError("Class has more than 1 label.")
            true_label = true_labels[0]

            # Get class index of max prob; exclude last column since it is label
            pred_class_index = np.argmax(row[: len(row) - 1])
            actual_class_index = self.class_labels.index(true_label)

            # Use class index as label
            predictions.append(pred_class_index)
            labels.append(actual_class_index)

        index_labels = list(range(len(self.class_labels)))
        cm = confusion_matrix(labels, predictions, labels=index_labels, normalize='true')
        return cm

    def plot_confusion_matrix(self, results):
        """
        Plot confusion matrix 
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """

        cm = self.compute_confusion_matrix(results)
        print("\nConfusion Matrix")
        print(cm)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        hm = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        indices = list(range(len(self.class_labels)))
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xlabel("Prediction", fontsize=10)
        plt.yticks(indices, self.class_labels, fontsize=10)
        plt.xticks(indices, self.class_labels, rotation=-90, fontsize=10)
        plt.colorbar(hm)
        thex_utils.display_and_save_plot(self.dir, "Confusion Matrix", fig=fig)

    def compute_performance(self, class_metrics):
        """
        Get recall, precision, and specificities per class.
        """
        precisions = {}
        recalls = {}
        specificities = {}  # specificity = true negative rate
        for class_name in class_metrics.keys():
            metrics = class_metrics[class_name]
            den = metrics["TP"] + metrics["FP"]
            precisions[class_name] = metrics["TP"] / den if den > 0 else 0
            den = metrics["TP"] + metrics["FN"]
            recalls[class_name] = metrics["TP"] / den if den > 0 else 0

            specificities[class_name] = metrics["TN"] / \
                (metrics["TN"] + metrics["FP"])
        return recalls, precisions, specificities

    def compute_confintvls(self, set_totals):
        """
        Compute 95% confidence intervals, [µ − 2σ, µ + 2σ],
        for each class
        :param set_totals: Map from fold # to map of metrics
        """

        print("Number of runs: " + str(self.num_runs))

        def get_cis(values, N):
            """
            Calculate confidence intervals [µ − 2σ, µ + 2σ] where 
            σ = sqrt( (1/ N ) ∑_n (a_i − µ)^2 )
            :param N: number of folds
            """
            mean = sum(values) / len(values)
            a = sum((np.array(values) - mean) ** 2)
            stdev = np.sqrt((1 / N) * a)
            return [mean - (2 * stdev), mean + (2 * stdev)]

        # 95% confidence intervals, [µ − 2σ, µ + 2σ]
        prec_cis = {cn: [0, 0] for cn in self.class_labels}
        recall_cis = {cn: [0, 0] for cn in self.class_labels}
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
            prec_cis[class_name] = get_cis(precisions, self.num_runs)
            recall_cis[class_name] = get_cis(recalls, self.num_runs)
        return prec_cis, recall_cis

    def plot_all_metrics(self, class_metrics, set_totals, y):
        """
        Plot performance metrics for model
        :param class_metrics: Returned from compute_metrics; Map from class name to map of performance metrics
        :param set_totals: Map from class name to map from fold # to map of metrics
        """
        recalls, precisions, specificities = self.compute_performance(class_metrics)
        pos_baselines, neg_baselines, precision_baselines = self.compute_baselines(
            self.class_counts, y)

        prec_intvls, recall_intvls = self.compute_confintvls(set_totals)

        self.plot_metrics(recalls, "Completeness", pos_baselines, recall_intvls)
        self.plot_metrics(
            specificities, "Completeness of Negative Class Presence", neg_baselines)
        self.plot_metrics(precisions, "Purity", precision_baselines, prec_intvls)

    def plot_metrics(self, class_metrics, xlabel, baselines=None, intervals=None):
        """
        Visualizes accuracy per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        :param xlabel: Label to assign to y-axis
        :[optional] param baselines: Mapping from class name to random-baseline performance
        # of samples in each class (get plotted atop the bars)
        :[optional] param annotations: List of
        """
        class_names, metrics, baselines, intervals = self.get_ordered_metrics(
            class_metrics,
            baselines,
            intervals)
        # Set constants
        tick_size = 8
        bar_width = 0.8
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, tight_layout=True)
        ax = plt.subplot()

        y_indices = np.arange(len(metrics))

        errs = None
        capsize = None
        if intervals is not None:
            errs = [[], []]
            for index, interval in enumerate(intervals):
                min_bar = interval[0]
                max_bar = interval[1]
                errs[0].append(metrics[index] - min_bar)
                errs[1].append(max_bar - metrics[index])

        # Plot metrics
        ax.barh(y=y_indices, width=metrics, height=bar_width,
                xerr=errs, capsize=2, ecolor='coral')

        # Plot random baselines
        if baselines is not None:
            for index, baseline in enumerate(baselines):
                plt.vlines(x=baseline, ymin=index - (bar_width / 2),
                           ymax=index + (bar_width / 2), linestyles='--', colors='red')

        # Format Axes, Labels, and Ticks
        ax.set_xlim(0, 1)
        plt.xticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.yticks(y_indices, class_names,  fontsize='small',
                   horizontalalignment='left')
        plt.ylabel('Transient Class', fontsize=10)
        max_tick_width = 0
        for i in class_names:
            bb = mpl.textpath.TextPath((0, 0), i, size=tick_size).get_extents()
            max_tick_width = max(bb.width, max_tick_width)
        yax = ax.get_yaxis()
        yax.set_tick_params(pad=max_tick_width + 2)

        ax.set_title(xlabel)
        thex_utils.display_and_save_plot(self.dir, self.name + ": " + xlabel)

    def get_ordered_metrics(self, class_metrics, baselines=None, intervals=None):
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

    def plot_prob_pr_curves(self, range_metrics, class_counts):
        """
        Plot recall curve and precision curve relative to probabilities
        :param range_metrics: Map of classes to [TP_range_sums, total_range_sums] where total_range_sums is the number of samples with probability in range for this class and TP_range_sums is the true positives per range
        :param class_counts: Map from class name to counts
        """

        for index, class_name in enumerate(range_metrics.keys()):
            true_positives, totals = range_metrics[class_name]
            fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            class_total = class_counts[class_name]
            if self.num_runs is not None:
                class_total = self.num_runs * class_total * .33

            precision = []  # Accuracy per range (true positive/total)
            recall = []
            TP_count = 0
            total_count = 0
            for index in reversed(range(len(true_positives))):
                cur_precision = 0
                cur_recall = 0

                TP_count += true_positives[index]
                total_count += totals[index]

                if total_count != 0:
                    # positive class samples / totals # with prob in range
                    cur_precision = TP_count / total_count
                if class_total != 0:
                    cur_recall = TP_count / class_total

                precision.append(cur_precision)
                recall.append(cur_recall)
            precision.reverse()
            recall.reverse()
            x_indices = np.linspace(0, 1, len(precision))

            color = 'tab:red'
            ax1.set_xlabel('Probability >=')
            ax1.set_ylabel('Purity', color=color)
            ax1.scatter(x_indices, precision, color=color, s=4)
            ax1.plot(x_indices, precision, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim([0, 1])

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Completeness', color=color)
            ax2.scatter(x_indices, recall, color=color, s=4)
            ax2.plot(x_indices, recall, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([0, 1])

            plt.title(class_name)

            thex_utils.display_and_save_plot(model_dir=self.dir,
                                             file_name=self.name + " Purity and Completeness vs. Probability: " + class_name,
                                             bbox_inches=None,
                                             fig=fig)

    def plot_probability_vs_class_rates(self, range_metrics):
        """
        Plots probability assigned to class (x-axis) vs the percentage of assignments that were that class (# of class A / all samples given probability of class in the range A).
        :param range_metrics: Map of classes to [TP_range_sums, total_range_sums] from compute_probability_range_metrics
        """

        perc_ranges = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        x_indices = np.arange(len(perc_ranges))
        total_count_per_range = np.zeros(10)
        for class_name in self.class_labels:
            true_positives, totals = range_metrics[class_name]

            pos_class_counts_per_range = np.array(self.class_positives[class_name])
            total_count_per_range += pos_class_counts_per_range

            prob_rates = self.class_prob_rates[class_name]
            print("\nProbability vs Class Rates for: " + str(class_name))
            print(prob_rates)
            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            ax.bar(x_indices, prob_rates)
            thex_utils.annotate_plot(ax, x_indices, prob_rates, totals)
            plt.xticks(x_indices, perc_ranges, fontsize=10)
            plt.yticks(list(np.linspace(0, 1, 11)), [
                       str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
            plt.xlabel('Probability of ' + class_name + ' +/- 5%', fontsize=12)
            plt.ylabel('Class Rate', fontsize=12)
            ax.set_title(class_name)
            thex_utils.display_and_save_plot(self.dir,
                                             "Probability vs Positive Rate: " + class_name)

        # Plot aggregated prob vs rates across all classes using weighted averages
        aggregated_rates = np.zeros(10)
        for class_name in self.class_labels:
            class_weights = np.array(self.class_positives[
                class_name]) / total_count_per_range
            pos_prob_rates = np.array(self.class_prob_rates[class_name])
            # Weighted rate = weight * rates
            aggregated_rates += np.multiply(class_weights, pos_prob_rates)
        print('\nAggregated Probability vs Class Rates')
        print(aggregated_rates)

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        ax.bar(x_indices, aggregated_rates)
        total = [int(b) for b in total_count_per_range]
        thex_utils.annotate_plot(ax, x_indices, aggregated_rates, total)
        plt.xticks(x_indices, perc_ranges, fontsize=10)
        plt.yticks(list(np.linspace(0, 1, 11)), [
            str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
        plt.xlabel('Probability of ' + class_name + ' +/- 5%', fontsize=12)
        plt.ylabel('Class Rate', fontsize=12)
        ax.set_title("Aggregated Probability vs Positive Rate")
        thex_utils.display_and_save_plot(self.dir,
                                         "Aggregated Probability vs Positive Rate")

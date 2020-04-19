
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

    def prep_err_bars(self, intervals, metrics):
        """
        Convert confidence intervals to specific values to be plotted
        """
        errs = [[], []]
        for index, interval in enumerate(intervals):
            min_bar = interval[0]
            max_bar = interval[1]
            errs[0].append(metrics[index] - min_bar)
            errs[1].append(max_bar - metrics[index])
        return errs

    def get_max_tick_width(self, class_names, tick_size):
        """
        Get the maximum tick width 
        """
        max_tick_width = 0
        for i in class_names:
            bb = mpl.textpath.TextPath((0, 0), i, size=tick_size).get_extents()
            max_tick_width = max(bb.width, max_tick_width)
        return max_tick_width + 2

    def get_avg_performances(self, unnorm_results):
        """
        Get average purity, completeness, and accuracy for each threshold of maintaing top i% maximum unnormalized probabilities 
        """
        # Init performance metrics
        p = []  # Avg purity for each i
        c = []  # Avg compelteness for each i
        a = []  # Avg accuracy for each i

        # 1. Get max density value per row
        probs_only = unnorm_results[:, 0:len(self.class_labels)].astype(float)
        max_value_per_row = np.amax(probs_only, axis=1)
        sorted_indices = np.flip(np.argsort(max_value_per_row))

        for i in np.linspace(0, 1, 100):
            # Get row indies of top i% of max_value_per_row
            top_i = int(len(sorted_indices) * i)
            top_i_indices = sorted_indices[:top_i]
            # Select rows at those indices
            top_i_densities = np.take(probs_only, indices=top_i_indices, axis=0)

            # Normalize these densities to compute metrics
            top_i_probs = top_i_densities / top_i_densities.sum(axis=1)[:, None]

            top_i_labels = np.take(unnorm_results,
                                   indices=top_i_indices,
                                   axis=0)[:, len(self.class_labels)]

            # Put probs & labels in same Numpy array
            i_results = np.hstack((top_i_probs, top_i_labels.reshape(-1, 1)))
            print("\n i results ")
            print(i_results)
            metrics, set_totals = self.compute_metrics(i_results, False)
            print("\nmetrics ")
            print(metrics)
            recalls, puritys, accs = self.compute_performance(metrics)
            print("\nrecalls ")
            print(recalls)
            avg_recall = sum(recalls.values()) / len(recalls)
            avg_purity = sum(puritys.values()) / len(puritys.values())
            avg_acc = sum(accs.values()) / len(accs.values())
            c.append(avg_recall)
            p.append(avg_purity)
            a.append(avg_acc)
        return p, c, a

    def plot_density_performance(self, unnorm_results):
        """
        Plots accuracy vs the X% of top unnormalized probabilities (densities) evaluated
        """
        print("Entering plot densityp erformance")
        p, c, a = self.get_avg_performances(unnorm_results)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                               dpi=DPI, tight_layout=True)
        ax.plot(np.linspace(0, 1, 100), p, color='tab:red')
        ax.set_ylabel("Average Purity", color='tab:red')
        ax.set_xlabel("% top densities")
        ax.set_ylim([0, 1])

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Average Completeness', color='tab:blue')
        ax2.plot(np.linspace(0, 1, 100), c, color='tab:blue')
        ax2.set_ylim([0, 1])

        thex_utils.display_and_save_plot(
            self.dir, "Average Purity and Completeness vs Density", fig=fig)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                               dpi=DPI, tight_layout=True)
        ax.plot(np.linspace(0, 1, 100), a)
        ax.set_ylabel("Average Accuracy")
        ax.set_xlabel("% top densities")
        ax.set_ylim([0, 1])
        thex_utils.display_and_save_plot(
            self.dir, "Average Accuracy")

    def compare_metrics(self, metrics_1, metrics_2, xlabel):
        """
        Compares performance between 2 sets of metrics.
        :param metrics_1, metrics_2: Mapping from class name to metric value. 1 is original and 2 is comparison.
        """
        class_names, mtrcs_1, b_1, intvls_1 = self.get_ordered_metrics(
            metrics_1,  None, None)
        class_names, mtrcs_2, b_2, intvls_2 = self.get_ordered_metrics(
            metrics_2,  None, None)

        # Set constants
        tick_size = 8
        bar_width = 0.4
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                               dpi=DPI, tight_layout=True)

        # Plot metrics
        y_indices = np.arange(len(mtrcs_1))
        ax.barh(y=y_indices, width=mtrcs_2,
                height=bar_width,  color='#99ffcc', label='Top 1/2')
        ax.barh(y=y_indices + bar_width, width=mtrcs_1, height=bar_width,
                color="#6699ff", label='Original')

        ax.legend()

        # Format Axes, Labels, and Ticks
        ax.set_xlim(0, 1)
        plt.xticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.yticks(y_indices, class_names,  fontsize='small',
                   horizontalalignment='left')
        plt.ylabel('Transient Class', fontsize=10)

        max_tick_width = self.get_max_tick_width(class_names, tick_size)
        yax = ax.get_yaxis()
        yax.set_tick_params(pad=max_tick_width)

        thex_utils.display_and_save_plot(
            self.dir, "Comparison_" + self.name + ": " + xlabel)

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
        accuracies = {}
        for class_name in class_metrics.keys():
            metrics = class_metrics[class_name]
            den = metrics["TP"] + metrics["FP"]
            precisions[class_name] = metrics["TP"] / den if den > 0 else 0
            den = metrics["TP"] + metrics["FN"]
            recalls[class_name] = metrics["TP"] / den if den > 0 else 0

            den = metrics["TP"] + metrics["TN"] + metrics["FP"] + metrics["FN"]
            accuracies[class_name] = ((
                metrics["TP"] + metrics["TN"]) / den) if den > 0 else 0
        return recalls, precisions, accuracies

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
        recalls, precisions, accuracies = self.compute_performance(class_metrics)
        pos_baselines, neg_baselines, precision_baselines = self.compute_baselines(
            self.class_counts, y)

        prec_intvls, recall_intvls = self.compute_confintvls(set_totals)

        self.plot_metrics(recalls, "Completeness", pos_baselines, recall_intvls)
        self.plot_metrics(precisions, "Purity", precision_baselines, prec_intvls)

    def plot_metrics(self, class_metrics, xlabel, baselines=None, intervals=None):
        """
        Visualizes metric per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        :param xlabel: Metric being plotted =  x-axis label
        :[optional] param baselines: Mapping from class name to random-baseline performance (get plotted atop the bars)
        :[optional] param intervals: confidence intervals, map from class 
        """
        class_names, metrics, baselines, intervals = self.get_ordered_metrics(
            class_metrics,
            baselines,
            intervals)
        # Set constants
        tick_size = 8
        bar_width = 0.8
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                               dpi=DPI, tight_layout=True)
        errs = self.prep_err_bars(intervals, metrics)
        y_indices = np.arange(len(metrics))
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

        max_tick_width = self.get_max_tick_width(class_names, tick_size)
        yax = ax.get_yaxis()
        yax.set_tick_params(pad=max_tick_width)

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
            thex_utils.annotate_plot(ax, x_indices, prob_rates,
                                     pos_class_counts_per_range)
            plt.xticks(x_indices, perc_ranges, fontsize=10)
            plt.yticks(list(np.linspace(0, 1, 11)), [
                       str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
            plt.xlabel('Probability of ' + class_name + ' +/- 5%', fontsize=12)
            plt.ylabel('Class Rate', fontsize=12)
            ax.set_title(class_name)
            thex_utils.display_and_save_plot(self.dir,
                                             "Probability vs Positive Rate: " + class_name)

        self.plot_agg_prob_vs_class_rates(total_count_per_range, True)

        self.plot_agg_prob_vs_class_rates(total_count_per_range, False)

    def get_agg_prob_vs_class_rates(self, total_count_per_range, weighted):
        """
        Get aggregated probability vs class rates
        """
        # Plot aggregated prob vs rates across all classes using weighted averages
        aggregated_rates = np.zeros(10)
        for class_name in self.class_labels:
            class_weights = np.array(self.class_positives[
                class_name]) / total_count_per_range
            pos_prob_rates = np.array(self.class_prob_rates[class_name])
            if weighted:
                # Weighted rate = weight * rates
                aggregated_rates += np.multiply(class_weights, pos_prob_rates)
            else:
                # Balanced average = (1/K) * rates
                aggregated_rates += (1 / len(self.class_labels)) * pos_prob_rates
        return aggregated_rates

    def plot_agg_prob_vs_class_rates(self, total_count_per_range, weighted):
        """
        :param total_count_per_range: Numpy array of length 10, with # of class samples in each range, total. So, last index is total number of samples with probability in range 90-100%
        """
        aggregated_rates = self.get_agg_prob_vs_class_rates(
            total_count_per_range, weighted)

        if weighted:
            p_title = 'Aggregated (Weighted) Probability vs Class Rates'
        else:
            p_title = 'Aggregated (Balanced) Probability vs Class Rates'

        print(p_title + ": \n" + str(aggregated_rates))

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        perc_ranges = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        x_indices = np.arange(len(perc_ranges))

        # Plot aggregated rates
        ax.bar(x_indices, aggregated_rates)
        total = [int(b) for b in total_count_per_range]
        thex_utils.annotate_plot(ax, x_indices, aggregated_rates, total)
        plt.xticks(x_indices, perc_ranges, fontsize=10)
        plt.yticks(list(np.linspace(0, 1, 11)), [
            str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
        plt.xlabel('Probability +/- 5%', fontsize=12)
        plt.ylabel('Class Rate', fontsize=12)
        ax.set_title(p_title)
        thex_utils.display_and_save_plot(self.dir, p_title)

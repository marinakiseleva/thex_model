
"""
Class Mixin for MainModel which contains all performance plotting functionality
"""
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mainmodel.helper_compute import *
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
        if intervals is None:
            return None
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

    def plot_example_output(self, row, i=None, priors=None):
        """
        Plots example output for a set of probabilities for a particular host-galaxy
        :param row: Numpy array of probabilities in order of self.class_labels and then TARGET_LABEL
        :param i: Index of sample
        :param priors: Boolean if using priors, for saving
        """
        labels = row[len(row) - 1]
        true_class_index = None
        for class_index, class_name in enumerate(self.class_labels):
            if class_name in thex_utils.convert_str_to_list(labels):
                true_class_index = class_index

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

        colors = ["#b3e0ff"] * len(self.class_labels)
        colors[true_class_index] = "#005c99"
        probabilities = row[0:len(row) - 1]
        x_indices = np.arange(len(self.class_labels))
        ax.bar(x_indices, probabilities, color=colors)
        ax.set_ylim([0, 1])
        plt.ylabel('Probability Assigned', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.xticks(x_indices, self.class_labels, fontsize=10)
        title = "example_output"
        if i is not None:
            title += "_" + str(i)
        if priors is not None:
            title += "_" + str(priors)
        thex_utils.display_and_save_plot(self.dir, title)

    def get_average(self, metrics):
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

    def get_avg_performances(self, unnorm_results):
        """
        Get average purity, completeness, and accuracy for each threshold of maintaing top i% maximum unnormalized probabilities
        """
        # Init performance metrics
        p = []  # Avg purity for each i
        c = []  # Avg compelteness for each i
        a = []  # Avg accuracy for each i
        a = []  # Avg accuracy for each i
        # 1. Get max density value per row
        probs_only = unnorm_results[:, 0:len(self.class_labels)].astype(float)
        max_value_per_row = np.amax(probs_only, axis=1)
        sorted_indices = np.flip(np.argsort(max_value_per_row))

        class_purities = {c: [] for c in self.class_labels}
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
            metrics, set_totals = self.compute_metrics(i_results, False)
            recalls, puritys = compute_performance(metrics)
            avg_recall = self.get_average(recalls)
            avg_purity = self.get_average(puritys)
            avg_acc = get_accuracy(metrics, N=top_i)

            c.append(avg_recall)
            p.append(avg_purity)
            a.append(avg_acc)
            class_purities = update_class_purities(class_purities, metrics, i)

        return p, c, a, class_purities

    def get_proportion_results(self, indices, results):
        """
        Reduce results (as probabilities) to those with these indices 
        :param indices: List of indices to keep 
        :param results: Results, with density per class and last column has label 
        """
        probs_only = results[:, 0:len(self.class_labels)].astype(float)

        # Select rows at those indices
        densities = np.take(probs_only, indices=indices, axis=0)

        # Normalize these densities to get probabilities
        probs = densities / densities.sum(axis=1)[:, None]

        labels = np.take(results,
                         indices=indices,
                         axis=0)[:, len(self.class_labels)]

        # Put probs & labels in same Numpy array
        prop_results = np.hstack((probs, labels.reshape(-1, 1)))
        return prop_results

    def plot_density_half_compare(self, unnorm_results):
        """
        Plot prob vs class rates for top 1/2 densities vs bottom 1/2
        """
        # Get top 1/2 densities (max density per row)
        probs_only = unnorm_results[:, 0:len(self.class_labels)].astype(float)
        max_value_per_row = np.amax(probs_only, axis=1)
        sorted_indices = np.flip(np.argsort(max_value_per_row))

        class_purities = {c: [] for c in self.class_labels}
        # Get row indies of top i% of max_value_per_row
        top_half = int(len(sorted_indices) * 0.5)
        top_half_indices = sorted_indices[:top_half]

        top_results = self.get_proportion_results(top_half_indices,  unnorm_results)
        top_metrics = self.compute_probability_range_metrics(
            top_results, bin_size=0.2, concat=False)
        print("\nVisualizing probability vs class rates for top 1/2 ")
        perc_ranges = [10, 30, 50, 70, 90]
        self.plot_probability_vs_class_rates(
            top_metrics, extra_title=" (top half)", perc_ranges=perc_ranges)

        bot_half_indices = sorted_indices[top_half:]
        bot_results = self.get_proportion_results(bot_half_indices,  unnorm_results)
        bot_metrics = self.compute_probability_range_metrics(
            bot_results, bin_size=0.2, concat=False)
        print("\nVisualizing probability vs class rates for bottom 1/2 ")
        self.plot_probability_vs_class_rates(
            bot_metrics, extra_title=" (bottom half)", perc_ranges=perc_ranges)

    def pre_plot_clean(self, x, y):
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

    def clean_plot(self, y, ax, name, color):
        """
        Helper plotting function for density analysis
        """
        orig_x = list(range(0, 100, 1))
        x, y = self.pre_plot_clean(orig_x,  y)
        print("\nPlotting " + str(name) + " versus % top densities. Y values:")
        print(y)
        ax.scatter(x, y, color=color, s=2)
        ax.plot(x, y, color=color, label=name)

    def plot_density_performance(self, unnorm_results):
        """
        Plots accuracy vs the X% of top unnormalized probabilities (densities) evaluated
        """
        p, c, a, class_purities = self.get_avg_performances(unnorm_results)

        # Plot aggregated performance metrics
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                               dpi=DPI, tight_layout=True)
        self.clean_plot(p, ax, "Purity", 'red')
        self.clean_plot(c, ax, "Completeness", 'blue')
        self.clean_plot(a, ax, "Accuracy", 'green')
        ax.set_xlabel("% Top Densities")
        ax.set_ylabel("Average %")
        ax.set_ylim([0, 1.01])
        ax.tick_params(axis='y')
        ax.legend()

        thex_utils.display_and_save_plot(
            self.dir, "Average Performance vs Density", fig=fig)

        # Plot purity per class
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                               dpi=DPI, tight_layout=True)

        colors = plt.get_cmap('tab20').colors
        for class_index, class_name in enumerate(self.class_labels):
            if self.num_runs is not None:
                raise ValueError(
                    "Can only aggregate this over folds, not runs.")
            total = self.class_counts[class_name]
            purities = class_purities[class_name]
            x = []
            y = []
            for vals in purities:
                if vals is not None:
                    TP, den, i, class_count = vals
                    x.append(i)
                    y.append(TP / den)
                    if round(class_count / total, 1) == 0.5:
                        star = [i, TP / den]
            print("\n" + class_name + " purities " + str([x for x in zip(x, y)]))
            ax.scatter(x, y, color=colors[class_index], s=2)
            ax.plot(x, y, color=colors[class_index], label=class_name)
            ax.plot(star[0], star[1], marker='*', color=colors[class_index])
        ax.set_ylabel("Purity")
        ax.set_xlabel("% Top Densities")
        ax.legend(loc='upper center', bbox_to_anchor=(1.25, 1), ncol=1, prop={'size': 8})

        thex_utils.display_and_save_plot(self.dir, "Prob Density % vs Purities")

    def plot_confusion_matrix(self, results):
        """
        Plot confusion matrix 
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """

        cm = compute_confusion_matrix(results, self.class_labels)
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

    def plot_all_metrics(self, class_metrics, set_totals, y):
        """
        Plot performance metrics for model
        :param class_metrics: Returned from compute_metrics; Map from class name to map of performance metrics
        :param set_totals: Map from class name to map from fold # to map of metrics
        """
        recalls, precisions = compute_performance(class_metrics)
        pos_baselines, neg_baselines, precision_baselines = self.compute_baselines(
            self.class_counts, y)

        prec_intvls, recall_intvls = compute_confintvls(
            set_totals, self.num_runs, self.class_labels)

        self.plot_metrics(recalls, "Completeness", pos_baselines, recall_intvls)
        self.plot_metrics(precisions, "Purity", precision_baselines, prec_intvls)
        print("\n\nCompleteness\n")
        print(str(recalls))
        print("\n\nPurity\n")
        print(str(precisions))

    def plot_metrics(self, class_metrics, xlabel, baselines=None, intervals=None):
        """
        Visualizes metric per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        :param xlabel: Metric being plotted =  x-axis label
        :[optional] param baselines: Mapping from class name to random-baseline performance (get plotted atop the bars)
        :[optional] param intervals: confidence intervals, map from class 
        """
        class_names, metrics, baselines, intervals = get_ordered_metrics(
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

    def plot_probability_vs_class_rates(self, range_metrics, extra_title="", perc_ranges=None):
        """
        Plots probability assigned to class (x-axis) vs the percentage of assignments that were that class (# of class A / all samples given probability of class in the range A). At top of each bar is AVERAGE  # assigned probability in that range (over runs), and bars are colored accordingly. If using cross-fold, count is just the total.
        :param range_metrics: Map of classes to [TP_range_sums, total_range_sums] from compute_probability_range_metrics
        :param extra_title: Extra string to add to title. 
        """
        if perc_ranges is None:
            perc_ranges = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        # Set +/- minus range based on number of xticks.
        if len(perc_ranges) == 10:
            pm = 5
        elif len(perc_ranges) == 5:
            pm = 10

        x_indices = np.arange(len(perc_ranges))
        # total_pos_pr - total positive # of samples per range.
        total_pos_pr = np.zeros(len(perc_ranges))
        for class_name in self.class_labels:
            # Collect data for this class
            true_positives, totals = range_metrics[class_name]
            pos_class_counts_per_range = np.array(self.class_positives[class_name])
            total_pos_pr += pos_class_counts_per_range
            prob_rates = self.class_prob_rates[class_name]

            if self.num_runs is not None:
                totals = [math.ceil(t / self.num_runs) for t in totals]

            # Plotting
            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            norm = plt.Normalize(0, max(totals))
            colors = mpl.cm.Blues(norm(totals))
            a = ax.bar(x_indices, prob_rates, color=colors, edgecolor='black')
            thex_utils.annotate_plot(ax, x_indices, prob_rates, totals)
            plt.xticks(x_indices, perc_ranges, fontsize=10)
            plt.yticks(list(np.linspace(0, 1, 11)), [
                       str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
            plt.xlabel('Probability of ' + class_name +
                       ' +/-' + str(pm) + '%', fontsize=12)
            plt.ylabel('Class Rate', fontsize=12)
            ax.set_title(class_name + extra_title)

            print("\nProbability vs Class Rates for: " + str(class_name))
            print(prob_rates)
            thex_utils.display_and_save_plot(self.dir,
                                             "Probability vs Positive Rate: " + class_name + extra_title)

        self.plot_agg_prob_vs_class_rates(total_pos_pr, True, perc_ranges)

        self.plot_agg_prob_vs_class_rates(total_pos_pr, False, perc_ranges)

    def plot_agg_prob_vs_class_rates(self, total_pos_pr, weighted, perc_ranges):
        """
        Aggregates probability versus class rates across all classes.
        :param total_pos_pr: Numpy array of length 10, with total # of positive class samples per range. So, last index is total number of TP samples with probability in range 90-100%
        :param weighted: Boolean to weigh by class frequency
        """
        aggregated_rates = get_agg_prob_vs_class_rates(
            total_pos_pr,
            self.class_labels,
            self.class_positives,
            self.class_prob_rates,
            weighted)

        if weighted:
            p_title = 'Aggregated (Weighted) Probability vs Class Rates'
        else:
            p_title = 'Aggregated (Balanced) Probability vs Class Rates'

        print(p_title + ": \n" + str(aggregated_rates))

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        x_indices = np.arange(len(perc_ranges))

        # Get average count per bin over runs
        if self.num_runs is not None:
            totals = [math.ceil(t / self.num_runs) for t in total_pos_pr]
        else:
            totals = [int(t) for t in total_pos_pr]

        norm = plt.Normalize(0, max(totals))
        colors = mpl.cm.Blues(norm(totals))

        # Plot aggregated rates
        ax.bar(x_indices, aggregated_rates, color=colors, edgecolor='black')

        thex_utils.annotate_plot(ax, x_indices, aggregated_rates, totals)
        plt.xticks(x_indices, perc_ranges, fontsize=10)
        plt.yticks(list(np.linspace(0, 1, 11)), [
            str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
        plt.xlabel('Probability +/- 5%', fontsize=12)
        plt.ylabel('Class Rate', fontsize=12)
        ax.set_title(p_title)
        thex_utils.display_and_save_plot(self.dir, p_title)

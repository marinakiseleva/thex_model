
"""
Class Mixin for MainModel which contains all performance plotting functionality
"""
import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mainmodel.helper_compute import *
from mainmodel.helper_plotting import *
from thex_data.data_consts import *
import utilities.utilities as thex_utils


class MainModelVisualization:
    """
    Mixin Class for MainModel performance visualization
    """

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
            if self.is_class(class_name, labels):
                true_class_index = class_index

        f, ax = plt.subplots(figsize=(5, 3), dpi=220)

        ACC = "#b3e0ff"  # actual class color, light blue
        DCC = "#005c99"  # default class color, dark blue

        colors = [DCC] * len(self.class_labels)
        colors[true_class_index] = ACC
        probabilities = row[0:len(row) - 1]
        # np.arange(len(self.class_labels))
        x_indices = np.linspace(0,
                                len(self.class_labels) * 0.4,
                                len(self.class_labels))
        ax.bar(x=x_indices, height=probabilities,  width=0.4, color=colors)
        ax.set_ylim([0, 1])
        plt.ylabel('Probability Assigned', fontsize=LAB_S)
        plt.xlabel('Class', fontsize=LAB_S)
        pretty_class_names = clean_class_name(self.class_labels)
        plt.xticks(x_indices, pretty_class_names, fontsize=TICK_S)
        title = "example_output"
        if i is not None:
            title += "_" + str(i)
        if priors is not None:
            title += "_" + str(priors)

        if not os.path.exists(self.dir + '/examples'):
            os.mkdir(self.dir + '/examples')
        thex_utils.display_and_save_plot(self.dir + '/examples', title)

    def get_avg_performances(self, unnorm_results):
        """
        Get average purity, completeness, and accuracy for each threshold of maintaing top i% maximum unnormalized probabilities
        """
        def update_class_purities(class_purities, class_metrics, i):
            """
            Helper function.
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
            metrics = self.compute_metrics(i_results)
            puritys, comps = get_puritys_and_comps(metrics)
            avg_comp = get_average(comps)
            avg_purity = get_average(puritys)
            avg_acc = get_accuracy(metrics, N=top_i)

            c.append(avg_comp)
            p.append(avg_purity)
            a.append(avg_acc)
            class_purities = update_class_purities(class_purities, metrics, i)

        return p, c, a, class_purities

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

        top_results = get_proportion_results(
            self.class_labels, top_half_indices,  unnorm_results)
        top_metrics = self.compute_probability_range_metrics(
            top_results, bin_size=0.2, concat=False)
        print("\nVisualizing probability vs class rates for top 1/2 ")
        perc_ranges = [10, 30, 50, 70, 90]
        self.plot_probability_vs_class_rates(
            top_metrics, extra_title=" (top half)", perc_ranges=perc_ranges)

        bot_half_indices = sorted_indices[top_half:]
        bot_results = get_proportion_results(
            self.class_labels, bot_half_indices,  unnorm_results)
        bot_metrics = self.compute_probability_range_metrics(
            bot_results, bin_size=0.2, concat=False)
        print("\nVisualizing probability vs class rates for bottom 1/2 ")
        self.plot_probability_vs_class_rates(
            bot_metrics, extra_title=" (bottom half)", perc_ranges=perc_ranges)

    def plot_density_performance(self, unnorm_results):
        """
        Plots accuracy vs the X% of top unnormalized probabilities (densities) evaluated
        """
        p, c, a, class_purities = self.get_avg_performances(unnorm_results)

        # Plot aggregated performance metrics
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                               dpi=DPI, tight_layout=True)
        clean_plot(p, ax, "Purity", 'red')
        clean_plot(c, ax, "Completeness", 'blue')
        clean_plot(a, ax, "Accuracy", 'green')
        ax.set_xlabel("% Top Densities", fontsize=LAB_S)
        ax.set_ylabel("Performance", fontsize=LAB_S)
        ax.set_ylim([0, 1.01])
        y_indices, y_ticks = get_perc_ticks()
        plt.xticks(y_indices * 100, y_ticks, fontsize=TICK_S)
        plt.yticks(y_indices, y_ticks, fontsize=TICK_S)
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
            star = None
            for vals in purities:
                if vals is not None:
                    TP, den, i, class_count = vals
                    x.append(i)
                    y.append(TP / den)
                    if round(class_count / total, 1) == 0.5:
                        star = [i, TP / den]
            pretty_class_name = clean_class_name(class_name)
            print("\n" + pretty_class_name + " purities " + str([x for x in zip(x, y)]))
            ax.scatter(x, y, color=colors[class_index], s=2)
            ax.plot(x, y, color=colors[class_index], label=pretty_class_name)
            if star is not None:
                ax.plot(star[0], star[1], marker='*', color=colors[class_index])

        ax.set_ylabel("Purity", fontsize=LAB_S)
        ax.set_xlabel("% Top Densities", fontsize=LAB_S)
        y_indices, y_ticks = get_perc_ticks()
        plt.xticks(y_indices, y_ticks, fontsize=TICK_S)
        plt.yticks(y_indices, y_ticks, fontsize=TICK_S)
        ax.legend(loc='upper center', bbox_to_anchor=(
            1.3, 1), ncol=1, prop={'size': LAB_S - 2})

        thex_utils.display_and_save_plot(self.dir, "Prob Density % vs Purities")

    def compute_metrics(self, results):
        """
        Compute TP, FP, TN, and FN per class
        :param results: List of 2D Numpy arrays with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :return class_metrics: Map from class name to map of {"TP": w, "FP": x, "FN": y, "TN": z}
        """
        class_metrics = {cn: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                         for cn in class_labels}
        for row in results:
            class_metrics = self.get_row_metrics(class_labels, row, class_metrics)
        return class_metrics

    def get_binary_row_metrics(self, row, class_metrics):
        """
        Get TP, FP, TN, FN metrics for this row & update in passed in dict of class_metrics.
        :param row: Numpy row of probabiliites (last col is label)
        :param class_metrics: Dict to update, of class names to map of metrics
        """
        # Last column is label
        label_index = len(self.class_labels)
        labels = row[label_index]
        for class_index, class_name in enumerate(self.class_labels):
            # Sample is an instance of this current class.
            prob = row[class_index]
            is_class = self.is_class(class_name, labels)
            if prob > 0.5 and is_class:
                class_metrics[class_name]["TP"] += 1
            elif prob > 0.5 and not is_class:
                class_metrics[class_name]["FP"] += 1
            elif prob <= 0.5 and is_class:
                class_metrics[class_name]["FN"] += 1
            elif prob <= 0.5 and not is_class:
                class_metrics[class_name]["TN"] += 1

        return class_metrics

    def get_row_metrics(self, row, class_metrics):
        """
        Get TP, FP, TN, FN metrics for this row & update in passed in dict of class_metrics.
        :param row: Numpy row of probabiliites (last col is label)
        :param class_metrics: Dict to update, of class names to map of metrics
        """
        if self.name == "Binary Classifiers":
            return self.get_binary_row_metrics(row, class_metrics)
        # Last column is label
        label_index = len(self.class_labels)
        labels = row[label_index]

        for class_index, class_name in enumerate(self.class_labels):

            # Get class index of max prob; exclude last column since it is label
            max_class_index = np.argmax(row[:len(row) - 1])
            max_class_name = self.class_labels[max_class_index]

            # Sample is an instance of this current class.
            is_class = self.is_class(class_name, labels)

            # class_name is actual class, and we got it as the max
            if is_class and max_class_name == class_name:
                class_metrics[class_name]["TP"] += 1
            # class_name is actual, but we didn't get it as max.
            elif is_class and max_class_name != class_name:
                class_metrics[class_name]["FN"] += 1
            # class_name is not actual, but we guessed it.
            elif not is_class and class_name == max_class_name:
                class_metrics[class_name]["FP"] += 1
            # class_name is not actual & we did not guess it
            elif not is_class and class_name != max_class_name:
                class_metrics[class_name]["TN"] += 1
        return class_metrics

    def get_pc_performance(self, pc_per_trial):
        """
        Get balanced purity, purity, and completeness, averaged over folds/trials. Return purity and completeness. (return balanced purity if flag is set)
        """
        N = self.num_runs if self.num_runs is not None else self.num_folds
        bps, ps, cs = self.get_avg_pc(pc_per_trial, N)

        thex_utils.pretty_print_dict(bps, self.name + " Balanced Purity")
        thex_utils.pretty_print_dict(ps, self.name + " Purity")
        thex_utils.pretty_print_dict(cs, self.name + " Completeness")
        if self.balanced_purity:
            ps = bps
        return ps, cs

    def get_pc_per_trial(self, results):
        """
        Get balanced purity, purity, and completeness per trial/fold, per class
        Return [[bp1,p1,c1], [bp2,p2,c2] ..., [bpN, pN, cN]] where each measure is a map from class names to value.
        """
        t_performances = []  # performance per trial/fold
        for index, trial in enumerate(results):
            class_metrics = {cn: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                             for cn in self.class_labels}
            # For each predicted row
            for row in trial:
                class_metrics = self.get_row_metrics(row, class_metrics)
            # Compute purity & completeness for this trial/fold (per class)

            puritys, comps = get_puritys_and_comps(class_metrics)

            bal_puritys = compute_balanced_purity(
                trial, self.class_labels, self.name)

            t_performances.append([bal_puritys, puritys, comps])
        return t_performances

    def get_avg_pc(self, t_performances, N):
        """
        Compute average balanced purity, normal purity, and completeness over folds/trials per class
        :param t_performances: Return of self.get_pc_per_trial
        :param N: Number of folds/trials
        """
        # Average purity & completeness over folds/trials (per class)
        mets = {v: {cn: 0 for cn in self.class_labels} for v in ["BP", "P", "C"]}
        for class_name in self.class_labels:
            # Counters for valid values of balanced purity and purity
            # purity could be None for fold with no TP.
            p_N = N  # N folds/trials
            bp_N = N
            # Aggregate over trials/folds
            for bp, p, c in t_performances:
                # No purity for this trial -> exclude from average
                if bp[class_name] is None:
                    bp_N = bp_N - 1
                else:
                    mets["BP"][class_name] += bp[class_name]
                if p[class_name] is None:
                    p_N = p_N - 1
                else:
                    mets["P"][class_name] += p[class_name]

                mets["C"][class_name] += c[class_name]
            # Average
            mets["BP"][class_name] = mets["BP"][class_name] / bp_N if bp_N > 0 else 0
            mets["P"][class_name] = mets["P"][class_name] / p_N if p_N > 0 else 0
            mets["C"][class_name] = mets["C"][class_name] / N if N > 0 else 0

        return mets["BP"], mets["P"], mets["C"]

    def plot_confusion_matrix(self, results):
        """
        Plot confusion matrix
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """
        cm = compute_confusion_matrix(results, self.class_labels)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        hm = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        indices = list(range(len(self.class_labels)))
        ax.set_ylabel("Actual", fontsize=LAB_S)
        ax.set_xlabel("Prediction", fontsize=LAB_S)
        pretty_class_names = clean_class_names(self.class_labels)
        plt.yticks(indices, pretty_class_names, fontsize=TICK_S)
        plt.xticks(indices, pretty_class_names, rotation=-90, fontsize=TICK_S)
        plt.colorbar(hm)
        print("\nConfusion Matrix")
        print(cm)
        thex_utils.display_and_save_plot(self.dir, "Confusion Matrix", fig=fig)

    def plot_all_metrics(self, purities, comps, all_pc, y):
        """
        Plot performance metrics for model
        :param purities: Average purity across folds/trials, per class (dict)
        :param comps: Average completeness across folds/trials, per class (dict)
        :param all_pc: Purity & completeness per trial/fold, per class
        :param y: all y dataset
        """
        a = self.get_num_classes()
        c_baselines, p_baselines = compute_baselines(
            self.class_counts,
            self.class_labels,
            self.get_num_classes(),
            self.balanced_purity,
            self.class_priors)
        p_intvls, c_intvls = compute_confintvls(
            all_pc, self.class_labels, self.balanced_purity)

        f, ax = plt.subplots(nrows=1, ncols=2,
                             sharex=True, sharey=True,
                             figsize=(8, 4),  dpi=DPI)

        self.plot_metrics_ax(ax[0], purities, "Purity", p_baselines, p_intvls)
        y_indices, class_names = self.plot_metrics_ax(
            ax[1], comps, "Completeness", c_baselines, c_intvls)

        plt.subplots_adjust(wspace=0, hspace=0)

        ax[0].set_yticks(y_indices)
        ax[0].set_yticklabels(clean_class_names(class_names),
                              fontsize=18,  horizontalalignment='right')

        thex_utils.display_and_save_plot(
            self.dir, self.name + "_metrics.pdf", bbox_inches='tight', fig=f)

    def plot_metrics_ax(self, ax, class_metrics, xlabel, baselines=None, intervals=None):
        """
        Visualizes metric per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        :param xlabel: Metric being plotted =  x-axis label
        :opt. param baselines: Mapping from class name to random-baseline performance (get plotted atop the bars)
        :opt. param intervals: confidence intervals, map from class [min, max]
        """
        # thex_utils.pretty_print_dict(class_metrics, xlabel + " per class")

        class_names, metrics, baselines, intervals = get_ordered_metrics(
            class_metrics,
            baselines,
            intervals)

        print("\n " + xlabel)
        thex_utils.pretty_print_mets(
            class_names, metrics, baselines, intervals)

        # Set constants
        bar_width = 0.4

        errs = prep_err_bars(intervals, metrics)
        max_y = (0.4 * len(metrics))
        if len(metrics) <= 2:
            max_y = max_y - 0.2
        y_indices = np.linspace(0, max_y, len(metrics))

        barcolor = P_BAR_COLOR if "Purity" in xlabel else C_BAR_COLOR
        # Plot bars
        ax.barh(y=y_indices, width=metrics, height=bar_width, xerr=errs,
                capsize=6,
                color=barcolor,
                edgecolor=BAR_EDGE_COLOR,
                ecolor=INTVL_COLOR)

        # Plot random baselines
        if baselines is not None:
            for index, baseline in enumerate(baselines):
                y_val = y_indices[index]
                ax.vlines(x=baseline,
                          ymin=y_val - (bar_width / 2),
                          ymax=y_val + (bar_width / 2),
                          linewidth=3,
                          linestyles=(0, (1, 1)), colors=BSLN_COLOR)

        # Format Axes, Labels, and Ticks
        ax.set_xlim(0, 1)
        x_indices, x_ticks = get_perc_ticks()
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_ticks, fontsize=TICK_S)
        ax.set_xlabel(xlabel + " (%)", fontsize=TICK_S)
        return y_indices, class_names

    def plot_prob_pc_curves(self, range_metrics):
        """
        Plot purity & completeness curves relative to >= probability assigned to event
        :param range_metrics: Map of classes to [TP_range_sums, total_range_sums] where total_range_sums is the number of samples with probability in range for this class and TP_range_sums is the true positives per range
        :param class_counts: Map from class name to counts
        """

        for index, class_name in enumerate(range_metrics.keys()):
            true_positives, totals = range_metrics[class_name]
            fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            class_total = self.class_counts[class_name]
            if self.num_runs is not None:
                class_total = self.num_runs * class_total * .33

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

            def plot_axis(ax, data, label, color):
                """
                Plot data on axis in certain color
                """
                x_indices = np.linspace(0, 1, len(data))
                ax.set_ylabel(label, color=color, fontsize=LAB_S)
                ax.scatter(x_indices, data, color=color, s=4)
                ax.plot(x_indices, data, color=color, linewidth=4)
                ax.set_ylim([0, 1])
                y_indices, y_ticks = get_perc_ticks()
                plt.yticks(ticks=y_indices, labels=y_ticks, color=color, fontsize=TICK_S)

            ax1.set_xlabel(r'Probability $\geq$', fontsize=LAB_S)
            plot_axis(ax1,  purities, "Purity", 'tab:red')
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            plot_axis(ax2,  comps, "Completeness", 'tab:blue')

            pretty_class_name = clean_class_name(class_name)
            x_indices, x_ticks = get_perc_ticks()
            plt.xticks(x_indices, x_ticks, fontsize=TICK_S)
            plt.title(pretty_class_name, fontsize=TITLE_S)

            thex_utils.display_and_save_plot(model_dir=self.dir,
                                             file_name=self.name + " Purity and Completeness vs. Probability: " + pretty_class_name,
                                             bbox_inches=None,
                                             fig=fig)

    def plot_probability_vs_class_rates(self, range_metrics, extra_title="", perc_ranges=None):
        """
        Plots probability assigned to class (x-axis) vs the percentage of assignments that were that class (# of class A / all samples given probability of class in the range A). At top of each bar is AVERAGE  # assigned probability in that range (over runs), and bars are colored accordingly. If using cross-fold, count is just the total.
        :param range_metrics: Map of classes to [TP_range_sums, total_range_sums] from compute_probability_range_metrics
        :param extra_title: Extra string to add to title.
        """
        if perc_ranges is None:
            perc_ranges = ["10%", "30%", "50%", "70%", "90%"]

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

            thex_utils.annotate_plot(
                ax=ax, x=x_indices, y=prob_rates, annotations=totals)
            plt.xticks(x_indices, perc_ranges, fontsize=TICK_S)
            y_indices, y_ticks = get_perc_ticks()
            plt.yticks(y_indices, y_ticks, fontsize=TICK_S)
            pretty_class_name = clean_class_name(class_name)
            plt.xlabel('Assigned Probability' + r' $\pm$' +
                       str(pm) + '%', fontsize=LAB_S)
            plt.ylabel('Empirical Probability', fontsize=LAB_S)
            ax.set_title(pretty_class_name + extra_title, fontsize=TITLE_S)
            m = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
            cbar = plt.colorbar(mappable=m)
            cbar.ax.tick_params(labelsize=LAB_S)

            print("\nProbability vs Class Rates for: " + str(pretty_class_name))
            print(prob_rates)
            thex_utils.display_and_save_plot(self.dir,
                                             "Probability vs Positive Rate: " + pretty_class_name + extra_title)

        # self.plot_agg_prob_vs_class_rates(total_pos_pr, True, perc_ranges)

        # self.plot_agg_prob_vs_class_rates(total_pos_pr, False, perc_ranges)

    def plot_agg_prob_vs_class_rates(self, total_pos_pr, weighted, perc_ranges):
        """
        Aggregates probability versus class rates across all classes.
        # of positive class samples per range. So, last index is total number of
        # TP samples with probability in range 90-100%
        :param total_pos_pr: Numpy array of length 10, with total
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
        plt.xticks(x_indices, perc_ranges, fontsize=TICK_S)

        y_indices, y_ticks = get_perc_ticks()
        plt.yticks(y_indices, y_ticks, fontsize=TICK_S)
        plt.xlabel('Probability ' + r'$\pm$' + '10%', fontsize=LAB_S)
        plt.ylabel('Empirical Probability', fontsize=LAB_S)
        ax.set_title(p_title, fontsize=TITLE_S, pad=20)
        m = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cbar = plt.colorbar(mappable=m)
        cbar.ax.tick_params(labelsize=LAB_S)

        thex_utils.display_and_save_plot(self.dir, p_title)

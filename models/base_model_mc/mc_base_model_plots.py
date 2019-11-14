import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc
from scipy import interp


from thex_data.data_clean import convert_class_vectors, relabel
from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, TARGET_LABEL, ROOT_DIR, UNDEF_CLASS, class_to_subclass

from sklearn.model_selection import StratifiedKFold


class MCBaseModelVisualization:
    """
    Mixin Class for Multiclass BaseModel performance visualization
    """

    def plot_probability_dists(self, class_probabilities, num_runs, k, X, y):
        """
        NEED TO UPDATE TO WORK WITH MULTIPLE RUNS.
        Plots the distribution of probabilities per class as scatter plot
        :param class_probabilities: List of Numpy matrices returned from get_all_class_probabilities for each run;
        each matrix has rows corresponding to samples, and each column the probability of that class
        """

        all_class_probs = np.concatenate((class_probabilities), axis=0)
        total_num_samples = np.shape(all_class_probs)[0]

        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
        test_indices = []
        for train_index, test_index in kf.split(X, y):
            test_indices.append(test_index)
        # df_indices: indices of y in order corresponding to all_class_probs
        df_indices = np.concatenate((test_indices), axis=0)

        mpl.rcParams['ytick.major.pad'] = '14'
        mpl.rcParams['ytick.minor.pad'] = '14'
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

        # columns in matrix are in order of self.class_labels
        for class_index, class_name in enumerate(self.class_labels):
            class_probs = all_class_probs[:, class_index]

            # is_class_probs: list of all probabilities assigned to TRUE samples of
            # this class
            is_class_probs = []
            # not_class_probs: list of all probs assigned to FALSE samples of this class
            not_class_probs = []
            class_count = 0
            for index, df_index in enumerate(df_indices):
                if class_name in y.iloc[df_index][TARGET_LABEL]:
                    is_class_probs.append(class_probs[index])
                    class_count += 1
                else:
                    not_class_probs.append(class_probs[index])
            # class_count *= num_runs

            x_vals = [class_index] * (total_num_samples - class_count)
            ax.scatter(not_class_probs, x_vals, s=2, c="red")

            x_vals = [class_index] * class_count

            ax.scatter(is_class_probs, x_vals, s=2, c="green")

        plt.yticks(np.arange(len(self.class_labels)),
                   self.class_labels, fontsize='xx-small')
        self.display_and_save_plot(
            title="Probability Distributions", ax=ax, bbox_inches=None, fig=fig)

    def plot_mc_probability_pos_rates(self, range_metrics):
        """
        Plots precision of class (y) vs. probability assigned to class (x)
        :param range_metrics: Map of classes to [percent_ranges, AP_range_sums, TOTAL_range_sums]
        """
        for index, class_name in enumerate(range_metrics.keys()):
            perc_ranges, AP_ranges, TOTAL_ranges = range_metrics[class_name]
            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

            perc_actual = []
            # annotations = [] Can be used to show X/Y instead of just Y
            for index, AP in enumerate(AP_ranges):
                p = 0
                total_predicted = TOTAL_ranges[index]
                if total_predicted != 0:
                    # Positive Class Samples / Total # with prob in range
                    p = AP / total_predicted
                # annotations.append(str(AP) + " / " + str(total_predicted))
                perc_actual.append(p)

            normalize = plt.Normalize(min(AP_ranges), max(AP_ranges))
            colors = plt.cm.Blues(normalize(AP_ranges))

            # Convert percent ranges to percents (orig decimals)
            perc_ranges = [int(i * 100) for i in perc_ranges]

            ax = self.plot_bar_with_annotations(
                axis=ax, x_vals=perc_ranges, y_vals=perc_actual, annotations=TOTAL_ranges, bar_colors=colors)
            plt.xlabel('Probability of ' + class_name + ' +/- 5%', fontsize=12)
            plt.ylabel('Purity', fontsize=12)
            self.display_and_save_plot("Probability vs Positive Rate: " + class_name, ax)

    def save_roc_curve(self, roc_plots, class_probabilities):
        """
        Plot ROC curve for each class, but do not show. Plot to axis attached to class's plot (saved in roc_plots). Used to save many curves for multiple runs.
        :param roc_plots: Mapping of class_name to [figure, axis, true positive rates, aucs]. This function plots curve on corresponding axis using this dict.
        :param class_probabilities: row for each sample, and each column is probability of class, in order of self.class_labels;  from get_all_class_probabilities
        """

        mean_fpr = np.linspace(0, 1, 100)
        y_test_vectors = convert_class_vectors(
            self.y_test, self.class_labels, self.class_levels, self.test_level)
        for class_index, class_name in enumerate(self.class_labels):
            f, ax, tprs, aucs = roc_plots[class_name]
            # preds: probability of each sample for this class_name
            preds = class_probabilities[:, class_index]
            labels = relabel(class_index, y_test_vectors)
            actual = np.transpose(labels[[TARGET_LABEL]].values)[0]
            fpr, tpr, thresholds = roc_curve(
                y_true=actual, y_score=preds, drop_intermediate=False)

            # Updates FPR and TPR to be on the range 0 to 1 (for plotting)
            # Python directly alters list objects within dict.
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0  # Start curve at 0,0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3)

        return roc_plots

    def plot_mc_roc_curves(self, roc_plots, num_folds):
        """
        Plots all ROC curves & average for each class separately, and plots the average ROC curves of each plot on 1 aggregated plot.
        :param roc_plots: :param roc_plots: Mapping from class_name to [figure, axis, true positive rates, aucs]. This function plots curve on corresponding axis using this dict; Results from save_roc_curve
        :param num_folds: Number of folds use in K-fold CV, used in title
        """
        avg_fig, avg_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        NUM_COLORS = len(self.class_labels)
        colors1 = plt.get_cmap('tab20b').colors
        colors2 = plt.get_cmap('tab20c').colors
        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
        cm = ListedColormap(colors)
        avg_ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS)
                                        for i in range(NUM_COLORS)])

        for class_name in self.class_labels:
            fig, ax, tprs, aucs = roc_plots[class_name]
            #   Baseline
            ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='r',
                    label='Baseline', alpha=.8)
            # Average line
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_fpr = np.linspace(0, 1, 100)
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC=%0.2f$\pm$%0.2f)' % (
                        mean_auc, std_auc),
                    lw=2, alpha=.8)
            avg_ax.plot(mean_fpr, mean_tpr, lw=1, alpha=0.6,
                        label=class_name + r' AUC=%0.2f$\pm$%0.2f' % (
                            mean_auc, std_auc))
            # Standard deviation in gray shading
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                            color='grey', alpha=.2, label=r'$\sigma$')
            title = class_name + " ROC Curve " + "over " + str(num_folds) + "-folds"
            ax.set_title(title)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="best")
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            self.save_plot(title=title, ax=ax,
                           bbox_inches=extent.expanded(1.3, 1.3), fig=fig)

        avg_ax.set_title(self.name + " ROC Curves")
        avg_ax.set_xlabel('False Positive Rate')
        avg_ax.set_ylabel('True Positive Rate')

        avg_ax.legend(loc='best',  prop={'size': 3})

        self.display_and_save_plot(title="ROC Summary", ax=avg_ax,
                                   bbox_inches=None, fig=avg_fig)

        # Save legend separately if classes < 20
        if len(self.class_labels) < 40:

            handles, labels = avg_ax.get_legend_handles_labels()
            legend_fig, legend_ax = plt.subplots(
                figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            legend = legend_ax.legend(handles, labels, loc='upper center')
            # Erase everything except the legend
            legend_ax.xaxis.set_visible(False)
            legend_ax.yaxis.set_visible(False)
            plt.show()

            legend_ax.axis('off')
            self.save_plot(title="Legend", ax=legend_ax,
                           bbox_inches='tight', fig=legend_fig, extra_artists=(legend,))

    def plot_mc_performance(self, class_metrics, xlabel, baselines=None):
        """
        Visualizes accuracy per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        :param xlabel: Label to assign to y-axis
        :[optional] param baselines: Mapping from class name to random-baseline performance
        # of samples in each class (get plotted atop the bars)
        :[optional] param annotations: List of
        """
        class_names, metrics, baselines = self.get_ordered_metrics(
            class_metrics, baselines)

        # Set constants
        tick_size = 8
        bar_width = 0.8
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, tight_layout=True)
        ax = plt.subplot()

        y_indices = np.arange(len(metrics))

        # Plot metrics
        ax.barh(y=y_indices, width=metrics, height=bar_width)

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
        plt.yticks(y_indices, class_names,  fontsize='xx-small',
                   horizontalalignment='left')
        plt.ylabel('Transient Class', fontsize=10)
        max_tick_width = 0
        for i in class_names:
            bb = mpl.textpath.TextPath((0, 0), i, size=tick_size).get_extents()
            max_tick_width = max(bb.width, max_tick_width)
        yax = ax.get_yaxis()
        yax.set_tick_params(pad=max_tick_width + 2)

        # if annotations is not None:
        #     self.add_bar_counts(y_indices, metrics, annotations, ax)
        self.display_and_save_plot(xlabel, ax)

    def get_classes_ordered(self, class_set, ordered_names):
        """
        Get list of class names ordered by place in hierarchy, so it would be for example: Ia, Ia 91BG, CC, etc. Properly ordered how we'd like them to show up in the plots.
        :param class_set: Set of classes to try adding to the list. We need to recurse on this list so subclasses are added right after parents. None at the start, and set manually to level 2 (top level after root)
        :param ordered_names: List of class names in correct, hierarchical order.
        """

        def get_ordered_intersection(level_classes):
            """
            Get classes at interesction between this hierarchy level and self.class_labels, while maintaining order given in hierarchy
            """
            intersection_classes = []
            for class_name in level_classes:
                if class_name in self.class_labels:
                    intersection_classes.append(class_name)
            return intersection_classes
        levels = list(self.level_classes.keys())[1:]
        if class_set is None:
            class_set = get_ordered_intersection(self.level_classes[2])

        for class_name in class_set:
            ordered_names.append(class_name)
            if class_name in class_to_subclass:
                subclasses = class_to_subclass[class_name]
                valid_subclasses = get_ordered_intersection(subclasses)
                if len(valid_subclasses) > 0:
                    ordered_names = self.get_classes_ordered(
                        valid_subclasses,
                        ordered_names)

        return ordered_names

    def get_ordered_metrics(self, class_metrics, baselines=None):
        """
        Reorder metrics and reformat class names in hierarchy groupings
        :param class_metrics: Mapping from class name to metric value.
        :[optional] param baselines: Mapping from class name to random-baseline performance
        """

        ordered_names = self.get_classes_ordered(None, [])
        ordered_formatted_names = []
        ordered_metrics = []
        ordered_baselines = [] if baselines is not None else None
        for class_name in ordered_names:
            # Add to metrics and baselines
            ordered_metrics.append(class_metrics[class_name])
            if baselines is not None:
                ordered_baselines.append(baselines[class_name])

            # Reformat class name based on depth and add to names
            pretty_class_name = class_name
            if UNDEF_CLASS in class_name:
                pretty_class_name = class_name.replace(UNDEF_CLASS, "")
                pretty_class_name = pretty_class_name + " (unspec.)"
            # level = self.class_levels[class_name]
            # if level > 2:
            #     indent = " " * (level - 1)
            #     symbol = ">"
            #     pretty_class_name = indent + symbol + pretty_class_name
            ordered_formatted_names.append(pretty_class_name)

        ordered_formatted_names.reverse()
        ordered_metrics.reverse()
        if baselines is not None:
            ordered_baselines.reverse()
        return [ordered_formatted_names, ordered_metrics, ordered_baselines]

    def compute_baselines(self, prior, class_counts, y):
        """
        """
        pos_baselines = {}
        neg_baselines = {}
        precision_baselines = {}

        total_count = y.shape[0]

        if prior == 'uniform':
            class_priors = {c: 1 / len(self.class_labels)
                            for c in self.class_labels}
        elif prior == 'frequency':
            class_priors = {c: class_counts[c] /
                            total_count for c in self.class_labels}
        else:
            raise ValueError("Priors not set.")

        for class_name in self.class_labels:
            # Compute baselines
            class_freq = class_counts[class_name] / total_count
            pos_baselines[class_name] = class_priors[class_name]
            neg_baselines[class_name] = (1 - class_priors[class_name])
            precision_baselines[class_name] = class_freq

        # TODO: need to recompute for level-based comparison.unclear if this is right.
        # elif normalized == 'level_based':
        #     # Calculate baselines per level of the class hierarchy.
        #     levels = list(self.level_classes.keys())[1:]
        #     for level_index, level in enumerate(levels):
        #         cur_level_classes = list(set(self.level_classes[level]).intersection(
        #             set(self.class_labels)))
        #         # Get level total
        #         total_count = sum(class_counts[c] for c in cur_level_classes)

        return pos_baselines, neg_baselines, precision_baselines

    def compute_performance(self, agg_metrics):
        """
        :param agg_metrics: Returned from aggregate_mc_class_metrics; Map from class name to map of performance metrics
        """
        precisions = {}
        recalls = {}
        specificities = {}  # specificity = true negative rate
        for class_name in agg_metrics.keys():
            metrics = agg_metrics[class_name]
            den = metrics["TP"] + metrics["FP"]
            precisions[class_name] = metrics["TP"] / den if den > 0 else 0
            den = metrics["TP"] + metrics["FN"]
            recalls[class_name] = metrics["TP"] / den if den > 0 else 0

            specificities[class_name] = metrics["TN"] / \
                (metrics["TN"] + metrics["FP"])
        return recalls, precisions, specificities

    def plot_metrics(self, agg_metrics, class_counts, prior, y):
        """
        Plot performance metrics for model
        :param agg_metrics: Returned from aggregate_mc_class_metrics; Map from class name to map of performance metrics
        """

        recalls, precisions, specificities = self.compute_performance(agg_metrics)
        pos_baselines, neg_baselines, precision_baselines = self.compute_baselines(
            prior, class_counts, y)

        self.plot_mc_performance(
            recalls, "Completeness", pos_baselines)
        self.plot_mc_performance(
            specificities, "Completeness of Negative Class Presence", neg_baselines)
        self.plot_mc_performance(precisions, "Purity", precision_baselines)

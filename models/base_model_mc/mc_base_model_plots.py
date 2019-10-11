import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp


from thex_data.data_clean import convert_class_vectors, relabel
from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, TARGET_LABEL, ROOT_DIR

from sklearn.model_selection import StratifiedKFold


class MCBaseModelVisualization:
    """
    Mixin Class for Multiclass BaseModel performance visualization
    """

    def plot_probability_dists(self, class_probabilities, k, X, y):
        """
        Plots the distribution of probabilities per class as scatter plot
        :param class_probabilities: List of Numpy matrices returned from get_all_class_probabilities for each run;
        each matrix has rows corresponding to samples, and each column the probability of that class
        """

        all_class_probs = np.concatenate((class_probabilities), axis=0)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        total_num_samples = np.shape(all_class_probs)[0]

        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
        test_indices = []
        for train_index, test_index in kf.split(X, y):
            test_indices.append(test_index)
        # df_indices: indices of y in order corresponding to all_class_probs
        df_indices = np.concatenate((test_indices), axis=0)

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

            x_vals = [class_index] * class_count
            ax.scatter(is_class_probs, x_vals, s=2, c="green")
            x_vals = [class_index] * (total_num_samples - class_count)
            ax.scatter(not_class_probs, x_vals, s=2, c="red")
        mpl.rcParams['ytick.major.pad'] = '4'
        plt.yticks(np.arange(len(self.class_labels)), self.class_labels)
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
            plt.ylabel('Class Presence Rate (Positive/Total)', fontsize=12)
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
        self.display_and_save_plot(title="ROC Summary", ax=avg_ax,
                                   bbox_inches=None, fig=avg_fig)

        # Save legend separately
        handles, labels = avg_ax.get_legend_handles_labels()
        legend_fig, legend_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        legend = legend_ax.legend(handles, labels, loc='upper center')
        # Erase everything except the legend
        legend_ax.xaxis.set_visible(False)
        legend_ax.yaxis.set_visible(False)
        plt.show()

        legend_ax.axis('off')
        self.save_plot(title="Legend", ax=legend_ax,
                       bbox_inches='tight', fig=legend_fig, extra_artists=(legend,))

    def plot_mc_performance(self, class_metrics, ylabel, base_lines=None, annotations=None):
        """
        Visualizes accuracy per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        :param ylabel: Label to assign to y-axis
        :[optional] param base_lines: Mapping from class name to random-baseline performance
        :[optional] param annotations: List of # of samples in each class (get plotted atop the bars)
        """
        class_names = list(class_metrics.keys())
        metrics = list(class_metrics.values())
        # Sort by class names, so the metrics, baselines, and annotations show up
        # consistently
        if base_lines is not None:
            base_lines = list(base_lines.values())
            if annotations is not None:
                class_names, metrics, base_lines, annotations = zip(
                    *sorted(zip(class_names, metrics, base_lines, annotations)))
            else:
                class_names, metrics, base_lines = zip(
                    *sorted(zip(class_names, metrics, base_lines)))
        elif annotations is not None:
            class_names, metrics, annotations = zip(
                *sorted(zip(class_names, metrics, annotations)))
        else:
            class_names, metrics = zip(*sorted(zip(class_names, metrics)))

        # Class names will be assigned in same order as these indices
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

        x_indices = np.arange(len(metrics))
        bar_width = 0.8
        ax.bar(x=x_indices, height=metrics, width=bar_width)
        ax.set_ylim(0, 1)
        plt.yticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
        plt.xticks(x_indices, class_names, fontsize=10, rotation=-90)
        plt.xlabel('Transient Class', fontsize=10)
        plt.ylabel(ylabel, fontsize=10)

        # Plot base lines
        # Map each level to the total # of classes at that level
        if base_lines is not None:
            # passed in specific baselines
            for index, baseline in enumerate(base_lines):
                plt.hlines(y=baseline, xmin=index - (bar_width / 2),
                           xmax=index + (bar_width / 2), linestyles='--', colors='red')

        if annotations is not None:
            self.add_bar_counts(x_indices, metrics, annotations, ax)
        self.display_and_save_plot(ylabel, ax)

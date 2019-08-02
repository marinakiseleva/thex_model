import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp


from thex_data.data_clean import convert_class_vectors, relabel
from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, TARGET_LABEL, ROOT_DIR


class MCBaseModelVisualization:
    """
    Mixin Class for Multiclass BaseModel performance visualization
    """

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

    def save_roc_curve(self, roc_plots):
        """
        Plot ROC curve for each class, but do not show. Plot to axis attached to class's plot (saved in roc_plots). Used to save many curves for multiple runs.
        :param roc_plots: Mapping of class_name to [figure, axis, true positive rates, aucs]. This function plots curve on corresponding axis using this dict.
        """
        mean_fpr = np.linspace(0, 1, 100)
        # class_probabilities has row for each sample, and each column is
        # probability of class, in order of self.class_labels
        class_probabilities = self.get_all_class_probabilities()
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
        Plot saved ROC curves
        :param roc_plots: Results from save_roc_curve
        """
        avg_fig, avg_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

        # Directory to save output in
        file_dir = ROOT_DIR + "/output/" + self.prep_file_name(self.name)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

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

            file_name = file_dir + "/" + self.prep_file_name(title)
            fig.savefig(file_name, bbox_inches=extent.expanded(1.3, 1.3))

        avg_ax.set_title(self.name + " ROC Curves")
        avg_ax.set_xlabel('False Positive Rate')
        avg_ax.set_ylabel('True Positive Rate')
        avg_ax.legend(loc="best", bbox_to_anchor=(1.2, 1))
        # Plot the mean of each class ROC curve on same plot
        avg_fig.savefig(file_dir + "/roc_summary",
                        bbox_inches=extent.expanded(1.6, 1.6))
        plt.show()

    def plot_mc_performance(self, class_metrics, ylabel, base_lines=None):
        """
        Visualizes accuracy per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        """
        class_names = list(class_metrics.keys())
        metrics = list(class_metrics.values())
        # Sort by class names, so they show up consistently
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
        if base_lines == True:
            level_counts = {}
            for class_name in class_names:
                level = self.class_levels[class_name]
                if level in level_counts:
                    level_counts[level] += 1
                else:
                    level_counts[level] = 1

            for index, class_name in enumerate(class_names):
                level = self.class_levels[class_name]
                # Random chance of class is 1/# of classes at this level
                level_base = 1 / level_counts[level]
                plt.hlines(y=level_base, xmin=index - 0.45, xmax=index +
                           bar_width - 0.45, linestyles='--', colors='red')
        elif base_lines is not None:
            # passed in specific baselines
            for index, class_name in enumerate(base_lines.keys()):
                baseline = base_lines[class_name]
                print("baseline for " + class_name + "  is " + str(baseline))
                plt.hlines(y=baseline, xmin=index - 0.45, xmax=index +
                           bar_width - 0.45, linestyles='--', colors='red')
        self.display_and_save_plot(ylabel, ax)

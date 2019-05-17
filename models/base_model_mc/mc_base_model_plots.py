from thex_data.data_clean import convert_class_vectors, relabel
from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, TARGET_LABEL
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp


class MCBaseModelVisualization:
    """
    Mixin Class for Multiclass BaseModel performance visualization
    """

    def plot_mc_probability_precision(self, range_metrics, title=None):
        """
        Plots precision of class (y) vs. probability assigned to class (x)
        :param range_metrics: Map of classes to [percent_ranges, AP_range_sums, TOTAL_range_sums]
        """
        for index, class_name in enumerate(range_metrics.keys()):
            perc_ranges, AP_ranges, TOTAL_ranges = range_metrics[class_name]
            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

            perc_actual = []
            annotations = []
            for index, AP in enumerate(AP_ranges):
                p = 0
                total_predicted = TOTAL_ranges[index]
                if total_predicted != 0:
                    # (TP) / (TP+FP)
                    p = AP / total_predicted
                annotations.append(str(AP) + " / " + str(total_predicted))
                perc_actual.append(p)

            normalize = plt.Normalize(min(AP_ranges), max(AP_ranges))
            colors = plt.cm.Blues(normalize(AP_ranges))

            ax = self.plot_bar_with_annotations(
                axis=ax, x_vals=perc_ranges, y_vals=perc_actual, annotations=TOTAL_ranges, bar_colors=colors)
            plt.xlabel('Probability of ' + class_name + ' +/- 5%', fontsize=12)
            plt.ylabel('AP/Total', fontsize=12)
            title = "Accuracy vs Probability" if title is None else title

            self.display_and_save_plot(title + ": " + class_name, ax)

    def save_roc_curve(self, i, roc_plots):
        """
        Plot ROC curve for each class, but do not show. Plot to axis attached to class's plot (saved in roc_plots). Used to save many curves for multiple runs.
        :param i: Fold or iteration number
        :param roc_plots: Mapping of class_name to [figure, axis, true positive rates, aucs]. This function plots curve on corresponding axis using this dict.
        """
        mean_fpr = np.linspace(0, 1, 100)
        # class_probabilities has row for each sample, and each column is
        # probability of class, in order of self.class_labels
        class_probabilities = self.get_all_class_probabilities()
        y_test_vectors = convert_class_vectors(
            self.y_test, self.class_labels, self.test_level)
        for class_index, class_name in enumerate(self.class_labels):
            f, ax, tprs, aucs = roc_plots[class_name]
            # column is the probability of each sample for this class_name
            column = class_probabilities[:, class_index]
            y_test_labels = np.transpose(relabel(class_index, y_test_vectors)[
                                         [TARGET_LABEL]].values)[0]
            fpr, tpr, thresholds = roc_curve(
                y_true=y_test_labels, y_score=column, drop_intermediate=False)

            # Updates FPR and TPR to be on the range 0 to 1 (for plotting)
            # Python directly alters list objects within dict.
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0  # Start curve at 0,0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3)
            # label='ROC fold %d (AUC=%0.2f)' % (i, roc_auc))

        return roc_plots

    def plot_mc_roc_curves(self):
        """
        Plot using Sklearn ROC Curve logic
        """
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        # class_probabilities is a numpy ndarray with a row for each X_test
        # sample, and a column for each class probability, in order of valid
        # self.class_labels
        class_probabilities = self.get_all_class_probabilities()
        # y_test_vectors has TARGET_LABEL column, with each class vector of length
        # self.class_labels
        y_test_vectors = convert_class_vectors(
            self.y_test, self.class_labels, self.test_level)
        for class_index, class_name in enumerate(self.class_labels):
            # If there is a valid model for this class
            column = class_probabilities[:, class_index]
            y_test_labels = np.transpose(relabel(class_index, y_test_vectors)[
                                         [TARGET_LABEL]].values)[0]
            fpr, tpr, thresholds = roc_curve(
                y_true=y_test_labels, y_score=column, sample_weight=None, drop_intermediate=True)

            plt.plot(fpr, tpr, label=class_name)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        # ax.plot(x, x, "--", label="Baseline")  # baseline
        plt.plot([0, 1], [0, 1], 'k--', label="Baseline")
        plt.title('ROC curve')
        plt.legend(loc='lower right')
        plt.show()

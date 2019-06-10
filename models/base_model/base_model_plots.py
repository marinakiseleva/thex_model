import os
import itertools
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from thex_data.data_consts import code_cat, ROOT_DIR, FIG_WIDTH, FIG_HEIGHT, DPI


class BaseModelVisualization:
    """
    Mixin Class for BaseModel performance visualization
    """

    def prep_file_name(self, text):
        """
        Remove unnecessary characters from text in order to save it as valid file name
        """
        replace_strs = ["\n", " ", ":", ".", ",", "/"]
        for r in replace_strs:
            text = text.replace(r, "_")
        return text

    def display_and_save_plot(self, title, ax):
        if ax is None:
            plt.title('\n'.join(wrap(title, 60)))
        else:
            ax.set_title('\n'.join(wrap(title, 60)))
        title = self.prep_file_name(title)
        plt.tight_layout()

        file_dir = ROOT_DIR + "/output/" + self.prep_file_name(self.name)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        plt.savefig(file_dir + "/" + title)

        plt.show()

    def plot_probability_precision(self, range_metrics, title=None):
        """
        Plots precision of class (y) vs. probability assigned to class (x)
        :param range_metrics: Map of classes to [percent_ranges, TP_ranges, AP_ranges, FP_ranges]
        """
        for index, class_code in enumerate(self.get_unique_classes()):
            perc_ranges, AP_ranges, TP_ranges, FP_ranges = range_metrics[class_code]
            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

            perc_correct = []
            for index, TP in enumerate(TP_ranges):
                p = 0
                total_predicted = TP + FP_ranges[index]
                if total_predicted != 0:
                    # TP/(TP+FP)

                    p = TP / (TP + FP_ranges[index])
                perc_correct.append(p)

            normalize = plt.Normalize(min(AP_ranges), max(AP_ranges))
            colors = plt.cm.Blues(normalize(AP_ranges))

            ax = self.plot_bar_with_annotations(
                axis=ax, x_vals=perc_ranges, y_vals=perc_correct, annotations=AP_ranges, bar_colors=colors)
            plt.xlabel('Probability of ' + code_cat[class_code] + ' +/- 5%', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            title = "Accuracy vs Probability" if title is None else title

            self.display_and_save_plot(title + ": " + str(code_cat[class_code]), ax)

    def plot_probability_completeness(self, range_metrics, title=None):
        """
        Plots recall of class (y) vs. probability assigned to class (x)
        :param range_metrics: Map of classes to [percent_ranges, TP_ranges, AP_ranges, FP_ranges]
        """
        for index, class_code in enumerate(self.get_unique_classes()):
            perc_ranges, AP_ranges, TP_ranges, FP_ranges = range_metrics[class_code]

            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

            perc_correct = []
            for index, corr in enumerate(TP_ranges):
                p = 0
                if AP_ranges[index] != 0:
                    # Correct predicitions / Actual count
                    p = corr / AP_ranges[index]
                perc_correct.append(p)

            normalize = plt.Normalize(min(AP_ranges), max(AP_ranges))
            colors = plt.cm.Blues(normalize(AP_ranges))

            ax = self.plot_bar_with_annotations(
                axis=ax, x_vals=perc_ranges, y_vals=perc_correct, annotations=AP_ranges, bar_colors=colors)
            plt.xlabel('Probability of ' + code_cat[class_code] + ' +/- 5%', fontsize=12)
            plt.ylabel('Recall', fontsize=12)
            title = "Accuracy vs Probability" if title is None else title

            self.display_and_save_plot(title + ": " + str(code_cat[class_code]), ax)

    def plot_probability_ranges(self, perc_ranges, y_values, ylabel, class_name, color_ranges, title=None):
        """
        Plots probability assigned to class (x) vs passed-in metric of y_values
        """
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        normalize = plt.Normalize(min(color_ranges), max(color_ranges))
        colors = plt.cm.Blues(normalize(color_ranges))
        ax = self.plot_bar_with_annotations(
            axis=ax, x_vals=perc_ranges, y_vals=y_values, annotations=color_ranges, bar_colors=colors)
        plt.xlabel('Probability of ' + class_name + ' +/- 5%', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        title = "Probability Assigned vs. " + ylabel if title is None else title
        self.display_and_save_plot(title + ": " + str(class_name), ax)

    def plot_roc_curves(self, rates, title=None):
        """
        Plot ROC curves of each class on same plot
        """
        if rates is None:
            raise ValueError("plot_roc_curves, rates cannot be None")

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        cm = plt.get_cmap('tab20c')
        NUM_COLORS = len(rates.keys())
        ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
        for index, class_code in enumerate(rates.keys()):
            FP_rates, TP_rates = rates[class_code]

            # Calculate area under curve
            auc = np.sum(TP_rates) / 100  # 100= # of values for x
            label_text = code_cat[class_code] + ", AUC: %.2f" % auc
            # Plotting ROC curve, FP against TP
            ax.plot(FP_rates, TP_rates, label=label_text)
        ax.grid()
        x = np.linspace(0, 1, num=100)
        ax.plot(x, x, "--", label="Baseline")  # baseline
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt_title = "ROC Curves" if title is None else title
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.legend(loc='best')
        self.display_and_save_plot(plt_title, ax)

    def plot_probability_metrics(self):
        """
        Plot ROC curves and distributions they are built on, for performance based on probabilities for each class
        :param test_data: Test data
        :param actual_classes: True labels for test set
        """
        unique_classes = self.get_unique_classes()

        f, ax = plt.subplots(len(unique_classes), 3,
                             figsize=(10, len(unique_classes) * 3), dpi=DPI)

        for index, class_code in enumerate(unique_classes):
            pos_probs, neg_probs = self.get_split_probabilities(
                class_code)

            class_name = code_cat[class_code]
            # Plot histogram of positive & negative probabilities
            self.plot_hists(pos_probs, neg_probs, ax[index, 0], class_name)

            # Plot PDF of positive & negative probabilities
            x, pos_pdf, neg_pdf = self.plot_pdfs(
                pos_probs, neg_probs, ax[index, 1], class_name)

            self.plot_roc(x, pos_pdf, neg_pdf, ax[index, 2])

        self.display_and_save_plot("probability metrics", ax=None)

    def plot_hists(self, pos_probs, neg_probs, ax, class_name):
        def plot_hist(samples, ax, color):
            """
            Plots histogram of samples on axis ax
            """
            ax.hist(x=samples, bins=10, range=(0, 1),
                    density=False, color=color, alpha=0.5)
            ax.set_xlim([0, 1])

        # Plot histogram of correct and incorrect probs
        plot_hist(pos_probs, ax, "green")
        plot_hist(neg_probs, ax, "red")
        ax.set_xlabel('P(class =' + class_name + ')', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('Distribution', fontsize=15)
        ax.legend(["Type " + class_name, "NOT Type " + class_name])

    def plot_pdfs(self, pos_probs, neg_probs, ax, class_name):
        """
        Plots the probability distribution function values of probabilities on ax
        """
        x, pos_pdf = self.get_normal_pdf(pos_probs)
        x, neg_pdf = self.get_normal_pdf(neg_probs)

        def plot_pdf(x, y, ax, color):
            ax.plot(x, y, color, alpha=0.5)
            ax.fill_between(x, y, facecolor=color, alpha=0.5)
            ax.set_xlim([0, 1])

        plot_pdf(x, pos_pdf, ax, color="green")
        plot_pdf(x, neg_pdf, ax, color="red")
        ax.set_xlabel('P(class =' + class_name + ')', fontsize=12)
        ax.set_ylabel('PDF', fontsize=12)
        ax.set_title('Probability Distribution of ' + class_name, fontsize=15)
        ax.legend(["Type " + class_name, "NOT Type " + class_name])
        return x, pos_pdf, neg_pdf

    def plot_roc(self, x, pos_pdf, neg_pdf, ax):
        """
        Plot ROC curve of x, given probabilities for positive and negative examples
        :param pos_pdf: Probability of class for positive examples
        :param neg_pdf: Probability of class for negative examples
        :param ax: axis to plot on
        """
        FP_rates, TP_rates = self.get_fp_tp_rates(x, pos_pdf, neg_pdf)
        # Plotting final ROC curve, FP against TP
        ax.plot(FP_rates, TP_rates)
        ax.plot(x, x, "--")  # baseline
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("ROC Curve", fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.grid()
        # auc = Probability that it will guess true over false
        auc = np.sum(TP_rates) / 100  # 100= # of values for x
        ax.legend(["AUC=%.3f" % auc])
        return ax

    def plot_confusion_matrix(self, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = confusion_matrix(
            self.y_test, self.predictions, labels=np.unique(self.y_test))

        classes = [code_cat[cc] for cc in np.unique(self.y_test)]

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix (without normalization)"

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        self.display_and_save_plot(title, ax)

    def plot_performance(self, class_metrics, plot_title, class_counts=None, ylabel="Accuracy", class_names=None):
        """
        Visualizes accuracy per class with bar graph
        :param class_metrics: Mapping from class codes to metric value.
        """

        # Get class names and corresponding accuracies
        if class_names is None:
            class_names = [code_cat[c] for c in class_metrics.keys()]
        metrics = [class_metrics[c] for c in class_metrics.keys()]

        # Sort by class names, so they show up consistently
        class_names, metrics = zip(*sorted(zip(class_names, metrics)))

        # Class names will be assigned in same order as these indices
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        ax = self.plot_bar_with_annotations(
            axis=ax, x_vals=class_names, y_vals=metrics, annotations=class_counts)
        plt.xticks(rotation=-90)
        plt.xlabel('Transient Class', fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        self.display_and_save_plot(plot_title, ax)

    #####################################
    # Plotting Utils ####################
    #####################################
    def plot_bar_with_annotations(self, axis, x_vals, y_vals, annotations, bar_colors=None):
        """
        Plots bar plot with annotations over each bar. Makes y-bar indices that map to percentages
        """
        x_indices = np.arange(len(y_vals))
        axis.bar(x_indices, y_vals, color=bar_colors)
        self.add_bar_counts(x_indices, y_vals, annotations, axis)
        plt.yticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=12)
        plt.xticks(x_indices, x_vals, fontsize=12)

    def add_bar_counts(self, x, y, class_counts, ax):
        """
        Adds count to top of each bar in bar plot
        """
        if class_counts is not None:
            class_index = 0
            for xy in zip(x, y):
                # Get class count of current class_index
                class_count = str(class_counts[class_index])
                ax.annotate(str(class_count), xy=xy,
                            textcoords='data', ha='center', va='bottom', fontsize=10)
                class_index += 1

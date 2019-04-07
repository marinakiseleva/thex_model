import itertools
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import pandas as pd
from sklearn.metrics import confusion_matrix

from thex_data.data_consts import code_cat, TARGET_LABEL, ROOT_DIR

FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 300


class BaseModelVisualization:
    """
    Mixin Class for BaseModel performance visualization
    """

    def display_and_save_plot(self, title, ax):
        if ax is None:
            plt.title('\n'.join(wrap(title, 60)))
        else:
            ax.set_title('\n'.join(wrap(title, 60)))

        replace_strs = ["\n", " ", ":", ".", ",", "/"]
        for r in replace_strs:
            title = title.replace(r, "_")
        model_dir = self.name.replace(" ", "_")
        plt.tight_layout()
        plt.savefig(ROOT_DIR + "/output/" + model_dir + "/" + title)

        plt.show()

    def plot_probability_correctness(self, prob_ranges=None, title=None):
        """
        Plots accuracy (y) vs. probability assigned to class (x)
        :param prob_ranges: [percent_ranges, corr_ranges, count_ranges]
        """

        X_accs = self.get_probability_matrix()
        # Add column of predicted class
        X_preds = pd.concat([X_accs, self.test_model()], axis=1)
        for index, class_code in enumerate(self.get_unique_classes()):
            if prob_ranges is None:
                perc_ranges, corr_ranges, count_ranges = self.get_corr_prob_ranges(
                    X_preds, class_code)
            else:
                perc_ranges, corr_ranges, count_ranges = prob_ranges[class_code]

            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

            perc_correct = []
            for index, corr in enumerate(corr_ranges):
                p = 0
                if count_ranges[index] != 0:
                    p = corr / count_ranges[index]
                perc_correct.append(p)

            normalize = plt.Normalize(min(count_ranges), max(count_ranges))
            colors = plt.cm.Blues(normalize(count_ranges))

            ax = self.plot_bar_with_annotations(
                axis=ax, x_vals=perc_ranges, y_vals=perc_correct, annotations=count_ranges, bar_colors=colors)
            plt.xlabel('Probability of ' + code_cat[class_code] + ' +/- 5%', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            title = "Accuracy vs Probability" if title is None else title

            self.display_and_save_plot(title + ": " + str(code_cat[class_code]), ax)

    def plot_roc_curves(self, rates=None, title=None):
        """
        Plot ROC curves of each class on same plot
        """

        unique_classes = self.get_unique_classes()

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS = len(unique_classes)
        ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
        for index, class_code in enumerate(unique_classes):
            if rates is None:
                FP_rates, TP_rates = self.get_roc_curve(class_code)

            else:
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

    def plot_accuracies(self, class_accuracies, plot_title, class_counts=None):
        """
        Visualizes accuracy per class with bar graph
        """

        # Get class names and corresponding accuracies
        class_names = [code_cat[c] for c in class_accuracies.keys()]
        accuracies = [class_accuracies[c] for c in class_accuracies.keys()]

        # Class names will be assigned in same order as these indices
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        ax = self.plot_bar_with_annotations(
            axis=ax, x_vals=class_names, y_vals=accuracies, annotations=class_counts)

        plt.xlabel('Transient Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
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

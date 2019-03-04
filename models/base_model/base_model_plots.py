import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from thex_data.data_consts import code_cat, TARGET_LABEL, ROOT_DIR

FIG_WIDTH = 6
FIG_HEIGHT = 4


class BaseModelVisualization:
    """
    Mixin Class for BaseModel performance visualization
    """

    def save_plot(self, title):
        replace_strs = ["\n", " ", ":", ".", ",", "/"]
        for r in replace_strs:
            title = title.replace(r, "_")
        model_dir = self.name.replace(" ", "_")
        plt.savefig(ROOT_DIR + "/output/" + model_dir + "/" + title)

    def plot_probability_correctness(self):
        """
        Plots accuracy (y) vs. probability assigned to class (x)
        """

        X_accs = self.get_probability_matrix()
        print(X_accs.dtypes)
        # Add column of predicted class
        # predictions = self.test_model()['predicted_class']
        X_preds = pd.concat([X_accs, self.test_model()], axis=1)
        print(list(X_preds))
        print(X_preds.dtypes)
        unique_classes = self.get_unique_test_classes()
        for index, class_code in enumerate(unique_classes):

            perc_ranges, perc_correct = self.get_corr_prob_ranges(
                X_preds, class_code)

            plt.bar(list(range(0, 10)), perc_correct)
            plt.xlabel('Probability of ' + code_cat[class_code] + ' +/- 5%', fontsize=12)
            plt.xlim([0, 1])
            plt.ylabel('Accuracy', fontsize=12)
            plt.yticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                        0.85, 0.95], perc_ranges, fontsize=10, rotation=30)
            plt.xticks(list(range(0, 10)), perc_ranges, fontsize=10, rotation=30)
            plt.title('Accuracy of ' + code_cat[class_code] +
                      ' Predictions by Probability Assigned', fontsize=15)
            plt.show()

    def plot_roc_curves(self):
        """
        Plot ROC curves of each class on same plot
        """

        unique_classes = self.get_unique_test_classes()

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
        cm = plt.get_cmap('gist_rainbow')
        NUM_COLORS = len(unique_classes)
        ax.set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
        for index, class_code in enumerate(unique_classes):
            pos_probs, neg_probs = self.get_split_probabilities(class_code)
            x, pos_pdf = self.get_normal_pdf(pos_probs)
            x, neg_pdf = self.get_normal_pdf(neg_probs)

            FP_rates, TP_rates = self.get_fp_tp_rates(x, pos_pdf, neg_pdf)

            # Calculate area under curve
            auc = np.sum(TP_rates) / 100  # 100= # of values for x
            label_text = code_cat[class_code] + ", AUC: %.2f" % auc
            # Plotting ROC curve, FP against TP
            ax.plot(FP_rates, TP_rates, label=label_text)
        ax.grid()
        ax.plot(x, x, "--", label="Baseline")  # baseline
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        title = "ROC Curves"
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.legend(loc='best')
        self.save_plot(title)
        plt.show()

    def plot_probability_metrics(self):
        """
        Plot ROC curves and distributions they are built on, for performance based on probabilities for each class
        :param test_data: Test data
        :param actual_classes: True labels for test set
        """
        unique_classes = self.get_unique_test_classes()

        f, ax = plt.subplots(len(unique_classes), 3,
                             figsize=(10, len(unique_classes) * 3), dpi=640)

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

        plt.tight_layout()
        self.save_plot("probability_metrics")
        plt.show()

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

    def get_normal_pdf(self, probabilities):
        """
        Returns normal PDF values
        """
        samples = np.array(probabilities)
        mean = np.mean(samples)
        std = np.sqrt(np.var(samples))
        x = np.linspace(0, 1, num=100)
        # Fit normal distribution to mean and std of data
        const = 1.0 / np.sqrt(2 * np.pi * (std**2))
        y = const * np.exp(-((x - mean)**2) / (2.0 * (std**2)))
        return x, y

    def get_fp_tp_rates(self, x, pos_pdf, neg_pdf):
        # Sum of all probabilities
        total_class = np.sum(pos_pdf)
        total_not_class = np.sum(neg_pdf)

        area_TP = 0  # Total area
        area_FP = 0  # Total area under incorrect curve

        TP_rates = []  # True positive rates
        FP_rates = []  # False positive rates
        # For each data point in x
        for i in range(len(x)):
            if pos_pdf[i] > 0:
                area_TP += pos_pdf[len(x) - 1 - i]
                area_FP += neg_pdf[len(x) - 1 - i]
            # Calculate FPR and TPR for threshold x
            # Volume of false positives over total negatives
            FPR = area_FP / total_not_class
            # Volume of true positives over total positives
            TPR = area_TP / total_class
            TP_rates.append(TPR)
            FP_rates.append(FPR)

        # Plotting final ROC curve, FP against TP
        return FP_rates, TP_rates

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

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
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
        plt.tight_layout()
        self.save_plot(title)
        plt.show()

    def plot_class_accuracy(self, plot_title, class_accuracies, class_counts=None):
        """
        Visualizes accuracy per class with bar graph
        """

        # Get class names and corresponding accuracies
        class_names = [code_cat[c] for c in class_accuracies.keys()]
        accuracies = [class_accuracies[c] for c in class_accuracies.keys()]

        # Class names will be assigned in same order as these indices
        class_indices = np.arange(len(class_accuracies.keys()))

        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=640)
        ax.bar(class_indices, accuracies)

        if class_counts is not None:
            cur_class = 0
            for xy in zip(class_indices, accuracies):
                # Get class count of current class_index
                cur_count = class_counts[cur_class]
                cur_class += 1
                ax.annotate(str(cur_count) + " total", xy=xy,
                            textcoords='data', ha='center', va='bottom', fontsize=12)
        plt.xlabel('Transient Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(class_indices, class_names, fontsize=12, rotation=30)
        plt.yticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=12)
        plt.title(plot_title, fontsize=15)
        self.save_plot(plot_title)
        plt.show()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

from models.base_model.base_model_performance import *

from thex_data.data_consts import code_cat, TARGET_LABEL, ROOT_DIR
from thex_data.data_plot import get_class_names


class KDEPerformance:

    def __init__(self, model):
        self.model = model

    def plot_probability_metrics(self):
        """
        Plot ROC curves and distributions they are built on, for performance based on predictions, for each class
        :param test_data: Test data
        :param actual_classes: True labels for test set
        """
        y_test = self.model.y_test[TARGET_LABEL]
        unique_classes = list(set(y_test))

        f, ax = plt.subplots(len(unique_classes), 3,
                             figsize=(10, len(unique_classes) * 3))
        class_prob_sums = self.get_class_prob_sums(self.model.X_test, unique_classes)

        for index, class_code in enumerate(unique_classes):
            pos_probs, neg_probs = self.get_split_probabilities(
                self.model.X_test.copy(), y_test, class_code, class_prob_sums)

            class_name = code_cat[class_code]
            # Plot histogram of positive & negative probabilities
            self.plot_hists(pos_probs, neg_probs, ax[index, 0], class_name)

            # Plot PDF of positive & negative probabilities
            x, pos_pdf, neg_pdf = self.plot_pdfs(
                pos_probs, neg_probs, ax[index, 1], class_name)

            self.plot_roc(x, pos_pdf, neg_pdf, ax[index, 2])

        plt.tight_layout()
        plt.show()

    def get_class_prob_sums(self, test_X, classes):
        """
        Get normalized probabilities and predictions -- normalizing over all predictions per class per sample
        """
        # Initialize dict from class to list of probabilities
        test_X = test_X.sample(frac=1)
        class_probs = {}
        class_prob_sums = {}
        class_counts = {}
        for c in classes:
            class_probs[c] = []
            class_prob_sums[c] = 0
            class_counts[c] = 0

        for index, row in test_X.iterrows():
            probabilities = self.model.calculate_class_probabilities(row)
            for c in classes:
                # if class_counts[c] < 20:
                class_probs[c].append(probabilities[c])
                class_counts[c] += 1

        for c in class_probs.keys():
            class_prob_sums[c] = sum(class_probs[c])
        # print(class_prob_sums)
        return class_prob_sums

    def normalize_probabilities(self, probabilities, class_prob_sums):
        for c in probabilities.keys():
            probabilities[c] = probabilities[c] / class_prob_sums[c]
        # Normalize all probabilities to sum to 1
        for c in probabilities.keys():
            probabilities[c] = probabilities[c] / sum(probabilities.values())
        return probabilities

    def get_split_probabilities(self, X_test, y_test, class_code, class_prob_sums):
        """
        Get probability assigned to the actual class per row and return probabilities for positive examples (pos_probs) and negative examples (neg_probs)
        """
        for index, row in X_test.iterrows():
            probabilities = self.model.calculate_class_probabilities(row)
            prob = probabilities[class_code]

            # Probability of this class for this row
            X_test.loc[index, 'probability'] = prob

            # Whether or not this data point IS this class
            actual_class = y_test.iloc[index]
            X_test.loc[index, 'is_class'] = True if (
                actual_class == class_code) else False

        pos_probs = X_test.loc[X_test.is_class == True]['probability']
        neg_probs = X_test.loc[X_test.is_class == False]['probability']

        return pos_probs, neg_probs

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
        x, pos_pdf = self.get_pdf(pos_probs)
        x, neg_pdf = self.get_pdf(neg_probs)

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

    def get_pdf(self, probabilities):
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

    def plot_roc(self, x, pos_pdf, neg_pdf, ax):
        """
        Plot ROC curve of x, given probabilities for positive and negative examples
        :param pos_pdf: Probability of class for positive examples
        :param neg_pdf: Probability of class for negative examples
        :param ax: axis to plot on
        """
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
        # auc = Probability that it will guess true over false
        auc = np.sum(TP_rates) / 100  # 100= # of values for x
        # Plotting final ROC curve, FP against TP
        ax.plot(FP_rates, TP_rates)
        ax.plot(x, x, "--")  # baseline
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("ROC Curve", fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.grid()
        ax.legend(["AUC=%.3f" % auc])

        return ax

"""
Independent Model
Assumes all classes are independent and are not related by any class hierarchy. 

"""

import numpy as np
import pandas as pd

from mainmodel.mainmodel import MainModel
from classifiers.optbinary import OptimalBinaryClassifier
import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS


class IndModel(MainModel):

    def __init__(self, **data_args):
        """
        """
        self.name = "Independent Model"
        super(IndModel, self).__init__(**data_args)

    def relabel_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = thex_utils.convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def train_model(self, X_train, y_train):
        """
        Train model using training data. All classes are a single disjoint set, so create a binary classifier for each one
        """

        self.models = {}
        for class_index, class_name in enumerate(self.class_labels):
            y_relabeled = self.relabel_class_data(class_name, y_train)

            print("\nClass Model: " + class_name)
            self.models[class_name] = OptimalBinaryClassifier(
                class_name, X_train, y_relabeled)

        return self.models

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            probabilities[class_name] = self.models[class_name].get_class_probability(x)
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = 0.001
                print("EnsembleModel get_class_probabilities NULL probability for " + class_name)

            if probabilities[class_name] < 0.0001:
                # Force min prob to 0.001 for future computation
                probabilities[class_name] = 0.001

        # Normalize
        probabilities = self.normalize(probabilities)

        return probabilities

    def normalize(self, probabilities):
        """
        Normalize across probabilities, treating each as independent. So, each is normalized by dividing by the sum of all probabilities.
        :param probabilities: Dict from class names to probabilities, already normalized across disjoint sets
        """
        total = sum(probabilities.values())
        norm_probabilities = {class_name: prob /
                              total for class_name, prob in probabilities.items()}

        return norm_probabilities

    def is_class(self, class_name, labels):
        """
        Boolean which returns True if class name is in the list of labels, and False otherwise.

        """
        if class_name in labels:
            return True
        else:
            return False

    def compute_probability_range_metrics(self, results, bin_size=.1):
        """
        Computes True Positive & Total metrics, split by probability assigned to class for ranges of 10% from 0 to 100. Used to plot probability assigned vs completeness (TP/total, per bin).
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :return range_metrics: Map of classes to [TP_range_sums, total_range_sums]
            total_range_sums: # of samples with probability in range for this class
            TP_range_sums: true positives per range 
        """
        range_metrics = {}
        label_index = len(self.class_labels)  # Last column is label
        for class_index, class_name in enumerate(self.class_labels):
            tp_probabilities = []  # probabilities for True Positive samples
            total_probabilities = []
            for result_set in results:
                for row in result_set:
                    labels = row[label_index]

                    # Sample is an instance of this current class.
                    is_class = self.is_class(class_name, labels)

                    # Get class index of max prob; exclude last column since it is label
                    max_class_prob = np.max(row[:len(row) - 1])
                    max_class_index = np.argmax(row[:len(row) - 1])
                    max_class_name = self.class_labels[max_class_index]

                    # tp_probabilities Numpy array of all probabilities assigned to this
                    # class that were True Positives
                    if is_class and max_class_name == class_name:
                        tp_probabilities.append(max_class_prob)

                    total_probabilities.append(row[class_index])

            # left inclusive, first bin is 0 <= x < .1. ; except last bin <=1
            bins = np.arange(0, 1 + bin_size, bin_size)
            tp_range_counts = np.histogram(tp_probabilities, bins=bins)[0].tolist()
            total_range_counts = np.histogram(total_probabilities, bins=bins)[0].tolist()

            range_metrics[class_name] = [tp_range_counts, total_range_counts]
        return range_metrics

    def compute_metrics(self, results):
        """
        Compute TP, FP, TN, and FN per class. Each sample is assigned its lowest-level class hierarchy label as its label. This is important, otherwise penalties will go across classes.
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :return class_metrics: Map from class name to map of {"TP": w, "FP": x, "FN": y, "TN": z}
        """

        # Combine sets in results
        results = np.concatenate(results)

        # Last column is label
        label_index = len(self.class_labels)
        class_metrics = {cn: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                         for cn in self.class_labels}

        for class_name in self.class_labels:
            for row in results:
                labels = row[label_index]
                # Sample is an instance of this current class.
                is_class = self.is_class(class_name, labels)
                # Get class index of max prob; exclude last column since it is label
                max_class_index = np.argmax(row[:len(row) - 1])
                max_class_name = self.class_labels[max_class_index]

                if class_name == max_class_name:
                    if is_class:
                        class_metrics[class_name]["TP"] += 1
                    else:
                        class_metrics[class_name]["FP"] += 1
                else:
                    if is_class:
                        class_metrics[class_name]["FN"] += 1
                    else:
                        class_metrics[class_name]["TN"] += 1

        return class_metrics

    def compute_baselines(self, class_counts, y):
        """
        Get random classifier baselines for recall, specificity (negative recall), and precision
        :param prior: NEED TO REINCORPORATE.
        """
        pos_baselines = {}
        neg_baselines = {}
        precision_baselines = {}

        total_count = y.shape[0]

        # if prior == 'uniform':
        class_priors = {c: 1 / len(self.class_labels)
                        for c in self.class_labels}
        # elif prior == 'frequency':
        #     class_priors = {c: class_counts[c] /
        #                     total_count for c in self.class_labels}

        for class_name in self.class_labels:
            # Compute baselines
            class_freq = class_counts[class_name] / total_count
            pos_baselines[class_name] = class_priors[class_name]
            neg_baselines[class_name] = (1 - class_priors[class_name])
            precision_baselines[class_name] = class_freq

        return pos_baselines, neg_baselines, precision_baselines

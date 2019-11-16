"""
Independent Model
Assumes all classes are independent and are not related by any class hierarchy.

"""

import numpy as np
import pandas as pd
from mainmodel.mainmodel import MainModel
from classifiers.optbinary import OptimalBinaryClassifier
import utilities.utilities as util
from thex_data.data_consts import TARGET_LABEL


class IndModel(MainModel):

    def __init__(self, **data_args):
        """
        """
        self.name = "Independent Model"
        super(IndModel, self).__init__(**data_args)

    def get_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = util.convert_str_to_list(row[TARGET_LABEL])
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
            y_relabeled = self.get_class_data(class_name, y_train)

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

    def get_label(self, labels):
        """
        Gets label at maximum depth in class hierarchy from string of labels.
        """
        labels = util.convert_str_to_list(labels)
        max_depth = 0
        max_label = None
        for label in labels:
            if label in self.class_levels and self.class_levels[label] > max_depth:
                max_depth = self.class_levels[label]
                max_label = label
        print('from all labels')
        print(labels)
        print("max label")
        print(max_label)
        return max_label

    def compute_metrics(self, results):
        """
        Compute TP, FP, TN, and FN per class. Each sample is assigned its lowest-level class hierarchy label as its label.
        """
        # Last column is label
        label_index = len(self.class_labels)
        class_metrics = {cn: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                         for cn in self.class_labels}
        for result_set in results:
            for row in result_set:
                print("for row ")
                print(row)

                # class_probability = row[class_index]
                labels = row[label_index]
                # Need to get single label by which to compare... Otherwise penalties
                # will go across classes.
                actual_label = self.get_label(labels)
                # Get class index of max prob; exclude last column since it is label
                max_class_index = np.argmax(row[:len(row) - 1])
                max_class_name = self.class_labels[max_class_index]
                print("Max Prob Class = Actual Class ?")
                print(str(max_class_name) + " = " + str(actual_label) + " ?")

                if max_class_name == actual_label:
                    # Correct prediction!
                    class_metrics[max_class_name]["TP"] += 1
                    print('this class is TP!')
                    for class_name in self.class_labels:
                        class_metrics[class_name]["TN"] += 1
                else:
                    # Incorrect prediction!
                    class_metrics[max_class_name]["FP"] += 1
                    class_metrics[actual_label]["FN"] += 1
                    for class_name in self.class_labels:
                        class_metrics[class_name]["TN"] += 1
        print(class_metrics)
        return class_metrics

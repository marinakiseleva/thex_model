from math import log
import pandas as pd
import numpy as np
from thex_data.data_clean import *
from thex_data.data_consts import *


class MCBaseModelPerformance:
    """
    Mixin Class for Multiclass BaseModel performance metrics. This is for subclasses that predict multiple classes at once, and are evaluated across all classes.
    """

    def get_mc_metrics_by_ranges(self, X_accs, class_name):
        """
        Collects metrics, split by probability assigned to class for ranges of 10% from 0 to 100. Used to plot probability assigned vs accuracy.
        :param X_accs: self.get_probability_matrix() output
        :return: [percent_ranges, AP_ranges, TOTAL_ranges] where
        percent_ranges: Range value
        AP_ranges: # of actual class_name in each range; TP + FN
        TOTAL_ranges: Total in each range
        """
        percent_ranges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        AP_ranges = []  # Positive class presence (Actually Positive)
        TOTAL_ranges = []  # Total in range (Positive and negative)
        for perc_range in percent_ranges:
            # Range of probabilities: + or - 5 from perc_range
            perc_range_min = perc_range - .05
            perc_range_max = perc_range + .05
            if perc_range_max == 1:
                perc_range_max += 1.01

            # Get all rows with probability of this class within range
            pred_col = class_name + '_prediction'
            actual_col = class_name + '_actual'
            X_in_range = X_accs.loc[(perc_range_min <= X_accs[pred_col]) & (
                X_accs[pred_col] < perc_range_max)]
            AP = X_in_range.loc[X_in_range[actual_col] == 1]

            AP_ranges.append(AP.shape[0])
            TOTAL_ranges.append(X_in_range.shape[0])

        for index, pr in enumerate(percent_ranges):
            percent_ranges[index] = str(int(pr * 100)) + "%"

        return [percent_ranges, AP_ranges, TOTAL_ranges]

    def get_mc_probability_matrix(self):
        """
        Gets probability of each class and actual class values, for each row
        classname1_prediction  classname1_actual  .. classnameN_prediction classnameN_actual
        .88           1         ...          .2              0
        """
        X_accuracy = pd.DataFrame()
        y_test_vectors = convert_class_vectors(
            self.y_test, self.class_labels, self.class_levels)
        for index, row in self.X_test.iterrows():
            probabilities = self.get_class_probabilities(row)
            target_classes = y_test_vectors.iloc[index][TARGET_LABEL]
            # Add each probability to dataframe
            for class_index, class_name in enumerate(probabilities.keys()):
                # Prediction is probability assigned to this class
                X_accuracy.loc[index, class_name +
                               '_prediction'] = probabilities[class_name]
                # Actual is 0 or 1 presence of this class
                X_accuracy.loc[index, class_name +
                               '_actual'] = target_classes[class_index]
        return X_accuracy

    def get_mc_unique_classes(self, df=None):
        """
        Get all unique class names from passed in DataFrame (or if None, self.y_test),
        and ensure they all exist in defined hierarchy
        :param df: Pandas DataFrame with TARGET_LABEL column with string list of classes per sample; if None self.y_test is used
        """
        if df is None:
            df = self.y_test
        unique_classes = []
        for index, row in df.iterrows():
            for label in convert_str_to_list(row[TARGET_LABEL]):
                if label != '':
                    unique_classes.append(label)
        unique_data_classes = set(unique_classes)

        # Ensure all classes are in defined hierarchy
        # Save all classes in hierarchy as set
        unique_defined_classes = []
        for parent in class_to_subclass.keys():
            children = class_to_subclass[parent]
            unique_defined_classes.append(parent)
            # Add unspecified parent class
            unique_defined_classes.append(UNDEF_CLASS + parent)
            unique_defined_classes.append(parent)
            for child in children:
                unique_defined_classes.append(child)

        unique_defined_classes = set(unique_defined_classes)

        classes = list(unique_data_classes.intersection(unique_defined_classes))
        classes.sort()

        # Exclude tree root
        classes.remove(TREE_ROOT)

        return classes

    def get_mc_class_metrics(self):
        """
        Save TP, FN, FP, TN, and BS(Brier Score) for each class.
        Brier score: (1 / N) * sum(probability - actual) ^ 2
        Log loss: -1 / N * sum((actual * log(prob)) + (1 - actual)(log(1 - prob)))
        self.y_test has TARGET_LABEL column with string list of classes per sample
        """

        # Relabel self.y_test actual labels with test_level label
        class_vectors = convert_class_vectors(self.y_test, self.class_labels,
                                              self.class_levels, self.test_level)
        # self.predictions is DataFrame with PRED_LABEL column of class name with
        # max probability
        predicted_classes = self.predictions
        actual_classes = self.y_test
        class_accuracies = {}

        for class_index, class_name in enumerate(self.class_labels):
            TP = 0  # True Positives
            FN = 0  # False Negatives
            FP = 0  # False Positives
            TN = 0  # True Negatives
            BS = 0  # Brier Score
            LL = 0  # Log Loss
            class_accuracies[class_name] = 0

            # predicted_classes has single class_name per row
            # class_one_zero has 1/0 class presence per row
            class_one_zero = relabel(class_index, class_vectors)

            # cur_level is hierarchical level of class_name
            cur_level = self.class_levels[class_name]

            for index, row in predicted_classes.iterrows():
                actual_label = class_one_zero.iloc[index][TARGET_LABEL]

                if self.test_level is not None:
                    predicted_label = predicted_classes.iloc[index][PRED_LABEL]
                else:
                    # Get class name with max probability at this level.
                    max_probability = 0
                    predicted_label = None
                    for cur_class in self.class_labels:
                        if self.class_levels[cur_class] == cur_level:
                            class_prob = predicted_classes.iloc[index][cur_class]
                            if class_prob > max_probability:
                                max_probability = class_prob
                                predicted_label = cur_class

                # Get probability of this class and clip to avoid log(0)
                prob = self.get_class_probabilities(
                    self.X_test.iloc[index])[class_name]
                e = 1e-13
                if prob == 0:
                    prob = e
                elif prob == 1:
                    prob = 1 - e

                if actual_label == 1:
                    LL += -log(prob)
                    if predicted_label == class_name:
                        TP += 1
                    else:
                        FN += 1
                elif actual_label == 0:
                    LL += -log(1 - prob)
                    if predicted_label == class_name:
                        FP += 1
                    else:
                        TN += 1
                BS += (prob - actual_label)**2

            BS /= self.y_test.shape[0]  # Brier Score
            LL /= self.y_test.shape[0]  # Divide Log Loss Sum by # of samples
            class_accuracies[class_name] = {"TP": TP,
                                            "FN": FN,
                                            "FP": FP,
                                            "TN": TN,
                                            "BS": BS,
                                            "LL": LL}

        return class_accuracies

    def aggregate_mc_class_metrics(self, metrics):
        """
        Aggregate list of class_accuracies from get_mc_class_metrics
        :param metrics: List of maps from class to stats, like: [{"Ia": {TP:3, FN:3, TN:4, ... }, "Ib":{TP: 3, FN: 3, TN: 4, ... }, ...}, {"Ia": {TP: 3, FN: 3, TN: 4, ... }, ...}, ... ]
        """
        agg_metrics = {class_name: {"TP": 0, "FN": 0, "FP": 0, "TN": 0,
                                    "BS": 0, "LL": 0} for class_name in self.class_labels}
        for metric_set in metrics:
            # metric_set is dictionary of class names to {TP: x, FN: y, TN: z, FN: d}
            for class_name in metric_set.keys():
                for metric in metric_set[class_name].keys():
                    agg_metrics[class_name][metric] += metric_set[class_name][metric]
            # Divide Brier Score and Log Loss sum by # of runs to get average
            agg_metrics[class_name]["BS"] /= len(metrics)
            agg_metrics[class_name]["LL"] /= len(metrics)
        return agg_metrics

    def aggregate_mc_prob_metrics(self, metrics):
        """
        Aggregate output of get_mc_metrics_by_ranges over several folds / runs: param metrics: Dictionary of class names to their ranged metrics,
         {class_name: [[percent_ranges, AP_ranges, TOTAL_ranges], ...], ...}
        return: Dictionary of class names to their SUMMED ranged metrics,
        {class_name: [percent_ranges, sum(AP_ranges), sum(TOTAL_ranges)], ...}
        """

        percent_ranges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        output_metrics = {}
        for class_name in metrics.keys():
            list_metrics = metrics[class_name]
            AP_range_sums = [0] * 10
            TOTAL_range_sums = [0] * 10
            for metric_response in list_metrics:
                # metric_response =[percent_ranges, AP_ranges, TOTAL_ranges]
                AP_range = metric_response[1]
                AP_range_sums = [sum(x) for x in zip(AP_range_sums, AP_range)]
                TOTAL_range = metric_response[2]
                TOTAL_range_sums = [sum(x) for x in zip(TOTAL_range_sums, TOTAL_range)]

            output_metrics[class_name] = [
                percent_ranges, AP_range_sums, TOTAL_range_sums]

        return output_metrics

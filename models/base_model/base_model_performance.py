import itertools
import collections
import numpy as np
import pandas as pd

from thex_data.data_consts import TARGET_LABEL, UNKNOWN_LABEL, PRED_LABEL, code_cat, cat_code


class BaseModelPerformance:
    """
    Mixin Class for BaseModel performance metrics
    """

    def get_metrics_by_ranges(self, X_accs, class_code):
        """
        Collects metrics, split by probability assigned to class for ranges of 10% from 0 to 100. Used to plot probability assigned vs completeness.
        :param X_accs: self.get_probability_matrix() output concated with predictions
        Returns [percent_ranges, AP_ranges, TP_ranges, FP_ranges] where
        percent_ranges: Range value
        AP_ranges: # of actual class_code in each range
        TP_ranges: # of correctly predicted class_code in each range
        FP_ranges: # of incorrectly predicted class_code in each range
        """
        class_code_str = str(class_code)
        percent_ranges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        AP_ranges = []  # Total count in each range
        TP_ranges = []  # Correctly predicted in each range
        FP_ranges = []  # False positives in each range
        for perc_range in percent_ranges:
            # Range of probabilities: + or - 5 from perc_range
            perc_range_min = perc_range - .05
            perc_range_max = perc_range + .05

            # Get all rows with probability of this class within range
            X_in_range = X_accs.loc[(perc_range_min <= X_accs[class_code_str]) & (
                X_accs[class_code_str] < perc_range_max)]

            AP = X_in_range.loc[(X_in_range.actual_class == class_code)]
            # FP is pred == class_code and actual != pred
            FP = X_in_range.loc[(X_in_range[PRED_LABEL] == class_code) & (
                X_in_range[PRED_LABEL] != X_in_range.actual_class)]
            TP = X_in_range.loc[(X_in_range[PRED_LABEL] == class_code) & (
                X_in_range[PRED_LABEL] == X_in_range.actual_class)]

            AP_ranges.append(AP.shape[0])
            TP_ranges.append(TP.shape[0])
            FP_ranges.append(FP.shape[0])

        for index, pr in enumerate(percent_ranges):
            percent_ranges[index] = str(int(pr * 100)) + "%"

        return [percent_ranges, AP_ranges, TP_ranges, FP_ranges]

    def get_probability_matrix(self, class_code=None):
        """
        Creates copy of X_test as DataFrame and includes probabilities and correctness of prediction. Without class_code it returns a DataFrame X_accuracy of the format:
        class1  class2  ... classN      actual_class
        .88        .1       .12             class2

        """
        X_accuracy = pd.DataFrame()
        if class_code is not None:
            # Get probabilities for this class alone
            for index, row in self.X_test.iterrows():
                probabilities = self.get_class_probabilities(row)

                # Probability of this class for this row
                X_accuracy.loc[index, 'probability'] = probabilities[class_code]

                # Whether or not this data point IS this class
                actual_class = self.y_test.iloc[index][TARGET_LABEL]
                X_accuracy.loc[index, 'is_class'] = True if (
                    actual_class == class_code) else False
                X_accuracy.loc[index, "actual_class"] = int(actual_class)
        else:
            """ Create matrix for all classes """
            unique_classes = self.get_unique_classes()
            for index, row in self.X_test.iterrows():
                probabilities = self.get_class_probabilities(row)
                # Add each probability to dataframe
                for c in unique_classes:
                    X_accuracy.loc[index, str(c)] = probabilities[c]
                actual_class = self.y_test.iloc[index][TARGET_LABEL]
                X_accuracy.loc[index, "actual_class"] = int(actual_class)

        return X_accuracy

    def get_split_probabilities(self, class_code):
        """
        Get probability assigned to the actual class per row and return probabilities for positive examples (pos_probs) and negative examples (neg_probs)
        """
        X_accuracy = self.get_probability_matrix(class_code)
        pos_probs = X_accuracy.loc[X_accuracy.is_class == True]['probability']
        neg_probs = X_accuracy.loc[X_accuracy.is_class == False]['probability']
        return pos_probs, neg_probs

    def get_roc_curve(self, class_code):
        """
        Returns false positive rates and true positive rates in order to plot ROC curve
        """
        pos_probs, neg_probs = self.get_split_probabilities(class_code)
        x, pos_pdf = self.get_normal_pdf(pos_probs)
        x, neg_pdf = self.get_normal_pdf(neg_probs)
        FP_rates, TP_rates = self.get_fp_tp_rates(x, pos_pdf, neg_pdf)

        return FP_rates, TP_rates

    def get_normal_pdf(self, probabilities):
        """
        Returns normal PDF values
        """
        samples = np.array(probabilities)
        mean = np.mean(samples)
        std = np.sqrt(np.var(samples))
        x = np.linspace(0, 1, num=100)
        # Fit normal distribution to mean and std of data
        if std == 0:
            const = 0
        else:
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

    def get_unique_classes(self, df=None):
        """
        Gets list of unique classes in testing set
        """
        df_classes = df[
            TARGET_LABEL] if df is not None else self.y_test[TARGET_LABEL]
        unique_classes = set(df_classes)

        # Add Unknown which is back-up assignment for samples with low probability
        unique_classes.add(cat_code[UNKNOWN_LABEL])

        return list(unique_classes)

    def get_class_counts(self, classes):
        """
        Gets the count of each existing class in this dataframe
        """
        df = self.combine_pred_actual()
        class_counts = []
        for tclass in classes:
            # Get count of this class
            class_count = df.loc[df[TARGET_LABEL] == tclass].shape[0]
            class_counts.append(class_count)
        return class_counts

    def get_class_performance(self, class_codes):
        """
        Record class performance by metrics that will later be used to compute precision and recall. Record: true positives, false positives, and # of Actual Positives, per class
        """
        df = self.combine_pred_actual()
        class_metrics = {}
        for class_code in class_codes:
            TP = df[(df[PRED_LABEL] == class_code) & (
                df[TARGET_LABEL] == class_code)].shape[0]
            FP = df[(df[PRED_LABEL] == class_code) & (
                df[TARGET_LABEL] != class_code)].shape[0]
            AP = df[df[TARGET_LABEL] == class_code].shape[0]
            class_metrics[class_code] = [AP, TP, FP]

        return class_metrics

    def get_recall(self, metrics, unique_classes):
        """
        Computes recall per class based on passed-in metrics
        :param metrics: Result of aggregate_metrics, {class1: [AP, TP, FP], ...}
        :return: mapping of class code to recall
        """
        recall = {}
        for class_code in metrics.keys():
            # Recall is TP/AP
            AP = metrics[class_code][0]
            TP = metrics[class_code][1]
            if AP == 0:
                recall[class_code] = 0
            else:
                recall[class_code] = TP / AP
        return recall

    def get_precision(self, metrics, unique_classes):
        """
        Computes precision per class based on passed-in metrics
        :param metrics: Result of aggregate_metrics, {class1: [AP, TP, FP], class2: [AP, TP, FP], ...}
        :return: mapping of class code to precision
        """
        precision = {}
        for class_code in metrics.keys():
            # Precision is TP/(# Positive Predicted) = TP/(TP+FP)
            TP = metrics[class_code][1]
            FP = metrics[class_code][2]
            if (TP + FP) == 0:
                precision[class_code] = 1
            else:
                precision[class_code] = TP / (TP + FP)

        return precision

    def combine_pred_actual(self):
        """
        Combines predicted with actual classes in 1 DataFrame with new 'correct' column which has a 1 if the prediction matches the actual class, and 0 otherwise
        """
        predicted_classes = self.predictions
        actual_classes = self.y_test

        if type(predicted_classes) == list:
            predicted_classes = pd.DataFrame(predicted_classes, columns=[PRED_LABEL])
        if type(actual_classes) == list:
            actual_classes = pd.DataFrame(actual_classes, columns=[TARGET_LABEL])

        # Reset index in order to ensure concat works properly
        predicted_classes.reset_index(drop=True, inplace=True)
        actual_classes.reset_index(drop=True, inplace=True)

        return pd.concat([predicted_classes, actual_classes], axis=1)

    ###########################
    # Aggregation Methods for Cross Fold Validation and Multiple Runs
    ###########################
    def aggregate_metrics(self, metrics, unique_classes):
        """
        Aggregates metrics (list of outputs from get_class_performance) by summing the TP, FP, and AP for each class across the maps in the list
        :param metrics: = [{class1 : [ranges, AP, TP, FP], class2: [ranges, AP, TP, FP], ...}, {}, ... ]
        Returns summed results:
        {class1: [ranges, AP, TP, FP], class2: [ranges, AP, TP, FP], ...}
        """
        summed_class_metrics = {cc: [0, 0, 0] for cc in unique_classes}
        for metric in metrics:
            for class_code in unique_classes:
                [AP, TP, FP] = metric[class_code]
                summed_class_metrics[class_code][0] += AP
                summed_class_metrics[class_code][1] += TP
                summed_class_metrics[class_code][2] += FP

        return summed_class_metrics

    def aggregate_range_metrics(self, prob_ranges):
        """
        Aggregates prob ranges over multiple runs
        :param prob_ranges: Mapping of classes to list of prob_ranges. in the format {class_code: [[percent_ranges,  AP_ranges, TP_ranges, FP_ranges] , ... ]  }
        Return prob_ranges in same format, but summed over runs
        """
        num_ranges = 10
        ranges = []
        for class_code in prob_ranges.keys():
            total_AP = [0] * num_ranges
            total_TP = [0] * num_ranges
            total_FP = [0] * num_ranges
            for set_of_rates in prob_ranges[class_code]:
                ranges = set_of_rates[0]  # stays same
                AP_ranges = set_of_rates[1]
                TP_ranges = set_of_rates[2]
                FP_ranges = set_of_rates[3]
                for index in range(num_ranges):
                    total_AP[index] += AP_ranges[index]
                    total_TP[index] += TP_ranges[index]
                    total_FP[index] += FP_ranges[index]

            prob_ranges[class_code] = [ranges, total_AP, total_TP, total_FP]
        return prob_ranges

    def aggregate_rocs(self, class_rocs):
        """
        Aggregates true positive & false positive rates in order to create averaged ROC curves over multiple folds/runs
        :param class_rocs: Mapping of class to rates {class: [[FP_rates, TP_rates], [FP_rates, TP_rates], ...]} where TP_rates and FP_rates are lists of 100 elements each 
        """

        len_X = 100  # Determined when creating PDFs
        for class_code in class_rocs.keys():
            sum_FP = [0] * len_X
            sum_TP = [0] * len_X
            for set_of_rates in class_rocs[class_code]:
                FP_rates = set_of_rates[0]
                TP_rates = set_of_rates[1]
                for index in range(len_X):
                    sum_TP[index] += TP_rates[index]
                    sum_FP[index] += FP_rates[index]

            # Compute average by dividing over total number of comparisons
            total = len(class_rocs[class_code])
            avg_FPs = [0] * len_X
            avg_TPs = [0] * len_X
            for index in range(len_X):
                avg_TPs[index] = sum_TP[index] / total
                avg_FPs[index] = sum_FP[index] / total
            # Reassign average TP and FP rates to this class
            class_rocs[class_code] = [avg_FPs, avg_TPs]
        return class_rocs

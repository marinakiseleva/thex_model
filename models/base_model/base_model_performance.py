import itertools
import collections
import numpy as np
import pandas as pd

from thex_data.data_consts import TARGET_LABEL

PRED_LABEL = 'predicted_class'


class BaseModelPerformance:
    """
    Mixin Class for BaseModel performance metrics
    """

    def get_corr_prob_ranges(self, X_accs, class_code):
        """
        Gets accuracy based on probability assigned to class for ranges of 5% from 0 to 100. Used to plot probability assigned vs probability of being correct
        Returns
        count_ranges: Count of total in each range 
        corr_ranges: Count of correctly predicted in each range
        percent_ranges: Ranges themselves
        """
        class_col = str(class_code)
        percent_ranges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        count_ranges = []  # Total count in each range
        corr_ranges = []  # Correctly predicted in each range
        for perc_range in percent_ranges:
            # Range of probabilities: + or - 5 from perc_range
            perc_range_min = perc_range - .05
            perc_range_max = perc_range + .05

            actual_in_range = X_accs.loc[(X_accs.actual_class == class_code) & (
                X_accs[class_col] >= perc_range_min) & (X_accs[class_col] < perc_range_max)]

            corr_pred_in_range = actual_in_range.loc[
                actual_in_range.predicted_class == class_code]

            corr_ranges.append(corr_pred_in_range.shape[0])
            count_ranges.append(actual_in_range.shape[0])

        for index, pr in enumerate(percent_ranges):
            percent_ranges[index] = str(int(pr * 100)) + "%"

        return percent_ranges, corr_ranges, count_ranges

    def get_probability_matrix(self, class_code=None):
        """
        Creates copy of X_test as DataFrame and includes probabilities and correctness of prediction
        """
        X_accuracy = self.X_test.copy()
        if class_code is not None:
            for index, row in X_accuracy.iterrows():
                probabilities = self.get_class_probabilities(row)
                prob = probabilities[class_code]

                # Probability of this class for this row
                X_accuracy.loc[index, 'probability'] = prob

                # Whether or not this data point IS this class
                actual_class = self.y_test.iloc[index][TARGET_LABEL]
                X_accuracy.loc[index, 'is_class'] = True if (
                    actual_class == class_code) else False
        else:
            """ Create matrix for all classes """
            unique_classes = self.get_unique_classes()
            for index, row in X_accuracy.iterrows():
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

        if df is not None:
            return list(set(df[TARGET_LABEL]))
        else:
            return list(set(self.y_test[TARGET_LABEL]))

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

    def get_class_precisions(self):
        """
        Get precision of each class separately
        """
        df = self.combine_pred_actual()
        class_codes = list(df[TARGET_LABEL].unique())
        class_precisions = {}
        for class_code in class_codes:
            class_precisions[class_code] = self.get_class_precision(df, class_code)
        return collections.OrderedDict(sorted(class_precisions.items()))

    def get_class_precision(self, df_compare, class_code):
        """
        Gets # of samples predicted correctly out of all positive predictions: # true positives/ # true positives + # of false positives
        """
        true_positives = df_compare[
            (df_compare[PRED_LABEL] == class_code) & (df_compare[TARGET_LABEL] == class_code)].shape[0]
        false_positives = df_compare[
            (df_compare[PRED_LABEL] == class_code) & (df_compare[TARGET_LABEL] != class_code)].shape[0]
        total = true_positives + false_positives
        if total == 0:
            precision = 0
        else:
            precision = true_positives / total
        return precision

    def get_class_recalls(self):
        """
        Get accuracy (recall) of each class separately
        """
        df = self.combine_pred_actual()
        class_codes = list(df[TARGET_LABEL].unique())
        class_recalls = {}
        for class_code in class_codes:
            class_recalls[class_code] = self.get_class_recall(df, class_code)
        return collections.OrderedDict(sorted(class_recalls.items()))

    def get_class_recall(self, df_compare, class_code):
        """
        Gets recall of this class: # true positives/ # of actual positives
        """
        true_positives = df_compare[
            (df_compare[PRED_LABEL] == class_code) & (df_compare[TARGET_LABEL] == class_code)].shape[0]
        actual_positives = df_compare[df_compare[TARGET_LABEL] == class_code].shape[0]
        recall = true_positives / actual_positives
        return recall

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

        df_compare = pd.concat([predicted_classes, actual_classes], axis=1)

        return df_compare

    ###########################
    # Aggregation Methods for Cross Fold Validation and Multiple Runs
    ###########################

    def aggregate_accuracies(self, model_results, unique_classes):
        """
        Aggregate class accuracies from several runs. Returns mapping of classes to average accuracy.
        :param model_results: List of accuracies from 1 or more runs; each item in list is a mapping from get_class_recalls
        :param y: Testing labels 
        """
        accuracy_per_class = {c: 0 for c in unique_classes}
        for class_accuracies in model_results:
            # class_accuracies maps class to accuracy as %
            for class_code in class_accuracies.keys():
                accuracy_per_class[class_code] += class_accuracies[class_code]
        # Divide each % by number of runs to get average accuracy
        return {c: acc / len(model_results) for c, acc in accuracy_per_class.items()}

    def aggregate_prob_ranges(self, prob_ranges):
        """
        Aggregates prob ranges over multiple runs
        :param prob_ranges: List of prob_ranges for multiple runs, in the format [percent_ranges, corr_in_range, count_in_range]
        :param k: Number of runs to average over
        Return prob_ranges in same format, but averaged over runs
        """
        num_ranges = 10
        percent_ranges = []
        for class_code in prob_ranges.keys():
            total_corr_in_range = [0] * num_ranges
            total_count_in_range = [0] * num_ranges
            for set_of_rates in prob_ranges[class_code]:
                percent_ranges = set_of_rates[0]  # stays same
                corr_in_range = set_of_rates[1]
                count_in_range = set_of_rates[2]
                for index in range(num_ranges):
                    total_corr_in_range[index] += corr_in_range[index]
                    total_count_in_range[index] += count_in_range[index]

            prob_ranges[class_code] = [percent_ranges,
                                       total_corr_in_range, total_count_in_range]
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

import itertools
import collections
import numpy as np
import pandas as pd

from thex_data.data_consts import TARGET_LABEL


class BaseModelPerformance:
    """
    Mixin Class for BaseModel performance metrics
    """

    def get_corr_prob_ranges(self, X_accs, class_code):
        """
        Gets accuracy based on probability assigned to class for ranges of 5% from 0 to 100. Used to plot probability assigned vs probability of being correct
        """
        class_col = str(class_code)
        percent_correct = []
        percent_ranges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        count_ranges = []  # Total correct count in each range
        for perc_range in percent_ranges:
            # Range of probabilities: + or - 5 from perc_range
            perc_range_min = perc_range - .05
            perc_range_max = perc_range + .05

            actual_in_range = X_accs.loc[(X_accs.actual_class == class_code) & (
                X_accs[class_col] >= perc_range_min) & (X_accs[class_col] < perc_range_max)]

            corr_pred_in_range = actual_in_range.loc[
                actual_in_range.predicted_class == class_code]

            if actual_in_range.shape[0] == 0:
                percent_correct.append(0)
                count_ranges.append(0)
            else:
                perc_correct = corr_pred_in_range.shape[0] / actual_in_range.shape[0]
                percent_correct.append(perc_correct)
                count_ranges.append(actual_in_range.shape[0])

        for index, pr in enumerate(percent_ranges):
            percent_ranges[index] = str(int(pr * 100)) + "%"
        return percent_ranges, percent_correct, count_ranges

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
        df = self.combine_dfs()
        class_counts = []
        for tclass in classes:
            # Get count of this class
            class_count = df.loc[df[TARGET_LABEL] == tclass].shape[0]
            class_counts.append(class_count)
        return class_counts

    def get_percent_correct(self, df_compare):
        """
        Gets % of rows of dataframe that have correct column marked as 1. This column indicates if TARGET_LABEL == predicted_class
        """
        count_correct = df_compare[df_compare.correct == 1].shape[0]
        count_total = df_compare.shape[0]
        perc_correct = count_correct / count_total
        return perc_correct

    def get_class_accuracies(self):
        """
        Get accuracy of each class separately
        """
        df = self.combine_dfs()
        class_codes = list(df[TARGET_LABEL].unique())
        class_accuracies = {}
        for class_code in class_codes:
            # filter df on this class code
            df_class = df[df[TARGET_LABEL] == class_code]
            class_accuracies[class_code] = self.get_percent_correct(df_class)
        return collections.OrderedDict(sorted(class_accuracies.items()))

    def combine_dfs(self):
        """
        Combines predicted with actual classes in 1 DataFrame with new 'correct' column which has a 1 if the prediction matches the actual class, and 0 otherwise
        """
        predicted_classes = self.predictions
        actual_classes = self.y_test
        PRED_LABEL = 'predicted_class'
        if type(predicted_classes) == list:
            predicted_classes = pd.DataFrame(predicted_classes, columns=[PRED_LABEL])
        if type(actual_classes) == list:
            actual_classes = pd.DataFrame(actual_classes, columns=[TARGET_LABEL])

        # Reset index in order to ensure concat works properly
        predicted_classes.reset_index(drop=True, inplace=True)
        actual_classes.reset_index(drop=True, inplace=True)

        df_compare = pd.concat([predicted_classes, actual_classes], axis=1)
        df_compare['correct'] = df_compare.apply(
            lambda row: 1 if row[PRED_LABEL] == row[TARGET_LABEL] else 0, axis=1)
        return df_compare

    ###########################
    # Aggregation Methods for Cross Fold Validation and Multiple Runs
    ###########################

    def aggregate_accuracies(self, model_results, unique_classes):
        """
        Aggregate accuracies from several runs. Returns mapping of classes to average accuracy.
        :param model_results: List of accuracies from 1 or more runs; each item in list is a mapping from get_class_accuracies
        :param y: Testing labels 
        """
        accuracy_per_class = {c: 0 for c in unique_classes}
        for class_accuracies in model_results:
            # class_accuracies maps class to accuracy as %
            for class_code in class_accuracies.keys():
                accuracy_per_class[class_code] += class_accuracies[class_code]
        # Divide each % by number of runs to get average accuracy
        return {c: acc / len(model_results) for c, acc in accuracy_per_class.items()}

    def aggregate_prob_ranges(self, prob_ranges, k):
        """
        Aggregate probability & correctness per range in order to plot aggregated version over multiple folds/runs
        :param prob_ranges: map of class_code : [percent_ranges, perc_correct, count_ranges]
        perc_correct = list, % of samples in each range correctly predicted
        count_ranges = list, # of samples in each range
        """
        # For each class
        num_ranges = 10
        totals_runs = len(prob_ranges.keys())
        percent_ranges = []
        for class_code in prob_ranges.keys():
            avg_perc_correct = [0] * num_ranges
            avg_count_in_range = [0] * num_ranges

            # set_of_rates = [percent_ranges, perc_correct, count_ranges]
            for set_of_rates in prob_ranges[class_code]:
                percent_ranges = set_of_rates[0]  # Does NOT need to be averaged
                perc_correct = set_of_rates[1]
                count_ranges = set_of_rates[2]

                for index in range(num_ranges):
                    avg_perc_correct[index] += perc_correct[index]
                    avg_count_in_range[index] += count_ranges[index]

            # Average each perc/count per range
            for index in range(num_ranges):
                avg_perc_correct[index] = int(avg_perc_correct[index] / k)
                avg_count_in_range[index] = int(avg_count_in_range[index] / k)

            prob_ranges[class_code] = [percent_ranges,
                                       avg_perc_correct, avg_count_in_range]
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

import itertools
import collections
import numpy as np
import pandas as pd

from thex_data.data_consts import TARGET_LABEL


class BaseModelPerformance:
    """
    Mixin Class for BaseModel performance metrics
    """

    def aggregate_accuracies(self, model_results, y):
        """
        Aggregate accuracies from several runs. Returns mapping of classes to average accuracy.
        :param model_results: List of accuracies from 1 or more runs; each item in list is a mapping from get_class_accuracies
        :param y: Testing labels 
        """
        accuracy_per_class = {c: 0 for c in y[TARGET_LABEL].unique().tolist()}
        for class_accuracies in model_results:
            # class_accuracies maps class to accuracy as %
            for tclass in class_accuracies.keys():
                accuracy_per_class[tclass] += class_accuracies[tclass]
        # Divide each % by number of folds to get average accuracy
        return {c: acc / len(model_results) for c, acc in accuracy_per_class.items()}

    def get_accuracy(self):
        """
        Returns overall accuracy of Naive Bayes classifier
        """
        perc_correct = get_percent_correct(self.combine_dfs())
        total_accuracy = round(perc_correct * 100, 4)
        return total_accuracy

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
        tclasses = list(df[TARGET_LABEL].unique())
        class_accuracies = {}
        for tclass in tclasses:
            df_ttype = df[df[TARGET_LABEL] == tclass]  # filter df on this ttype
            class_accuracies[tclass] = self.get_percent_correct(df_ttype)
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

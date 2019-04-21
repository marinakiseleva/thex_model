import numpy as np
import pandas as pd

from thex_data.data_transform import convert_class_vectors
from thex_data.data_consts import TARGET_LABEL, UNKNOWN_LABEL, PRED_LABEL, code_cat, cat_code


class MCBaseModelPerformance:
    """
    Mixin Class for Multiclass BaseModel performance metrics. This is for subclasses that predict multiple classes at once, and are evaluated across all classes. 
    """

    def get_mc_metrics_by_ranges(self, X_accs, class_name):
        """
        Collects metrics, split by probability assigned to class for ranges of 10% from 0 to 100. Used to plot probability assigned vs accuracy.
        :param X_accs: self.get_probability_matrix() output concated with predictions
        Returns [percent_ranges, AP_ranges, TOTAL_ranges] where
        percent_ranges: Range value
        AP_ranges: # of actual class_name in each range
        TOTAL_ranges: Total in each range
        """
        percent_ranges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        AP_ranges = []  # Positive class presence (Actually Positive)
        TOTAL_ranges = []  # Total in range (Positive and negative)
        for perc_range in percent_ranges:
            # Range of probabilities: + or - 5 from perc_range
            perc_range_min = perc_range - .05
            perc_range_max = perc_range + .05

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
        class1_prediction  class1_actual  .. classN_prediction classN_actual   
        .88                 1                   .2              0
        ...
        """
        X_accuracy = pd.DataFrame()
        y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)

        for index, row in self.X_test.iterrows():
            probabilities = self.get_class_probabilities(row)
            target_classes = y_test_vectors.iloc[index][TARGET_LABEL]
            # Add each probability to dataframe
            for class_index, class_name in enumerate(probabilities.keys()):
                X_accuracy.loc[index, class_name +
                               '_prediction'] = probabilities[class_name]
                X_accuracy.loc[index, class_name +
                               '_actual'] = target_classes[class_index]
        return X_accuracy

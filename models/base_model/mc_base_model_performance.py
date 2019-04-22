import pandas as pd

from thex_data.data_clean import convert_class_vectors, convert_str_to_list
from thex_data.data_consts import *


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

    def get_mc_unique_classes(self):
        """
        Gets all unique classes based for HMC model. In HMC, we save class vectors, so we need to figure out all the unique subclasses from these vectors.
        """
        unique_classes = []
        for index, row in self.y_test.iterrows():
            for label in convert_str_to_list(row[TARGET_LABEL]):
                unique_classes.append(label)
        return list(set(unique_classes))

    def get_mc_recall_scores(self):
        """
        Get recall of each class; returns map of class code to recall
        """
        all_class_recalls = {label: 0 for label in self.class_labels}
        for class_index, class_name in enumerate(self.class_labels):
            all_class_recalls[class_name] = self.get_mc_class_recall(class_index)
        class_recalls = {}
        unique_classes = self.get_mc_unique_classes()
        for class_name in all_class_recalls.keys():
            if class_name in unique_classes:
                class_code = cat_code[class_name]
                class_recalls[class_code] = all_class_recalls[class_name]
        return class_recalls

    def get_mc_class_recall(self, class_index):
        """
        Get recall of single class, of class_index. Recall is TP/TP+FN.
        """
        threshold = 0.5
        y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)
        row_count = y_test_vectors.shape[0]
        TP = 0  # True positive count, predicted = actual = 1
        FN = 0  # False negative count, predicted = 0, actual = 1
        for sample_index in range(row_count - 1):

            # Compare this index of 2 class vectors
            predicted_class = self.predictions[sample_index, class_index]

            actual_classes = y_test_vectors.iloc[sample_index][TARGET_LABEL]
            actual_class = actual_classes[class_index]

            if actual_class >= threshold:
                if actual_class == predicted_class:
                    TP += 1
                elif actual_class != predicted_class:
                    FN += 1
        denominator = TP + FN
        class_recall = TP / (TP + FN) if denominator > 0 else 0
        return class_recall

import pandas as pd
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
        class1_prediction  class1_actual  .. classN_prediction classN_actual
        .88                 1                   .2              0
        ...
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
        Gets all unique class names based for HMC model. In HMC, we save class vectors, so we need to figure out all the unique subclasses from these vectors.
        """
        if df is None:
            df = self.y_test
        unique_classes = []
        for index, row in df.iterrows():
            for label in convert_str_to_list(row[TARGET_LABEL]):
                if label != '':
                    unique_classes.append(label)
        return list(set(unique_classes))

    # def combine_mc_pred_actual(self, class_names):
    #     """
    #     Create DataFrame by which to gauge performance. Each class name is a column, which a 0 or 1 for actual class presence. Class is marked as UNDEFINED_ if the max label has an UNDEFINED_label in class_names. PRED_LABEL column contains current single-class prediction.
    #     :param class_names: List of classes to consider. Will be column for each class in output DataFrame with 1 if that sample row actually has that class and 0 otherwise.
    #     :return: DataFrame with PRED_LABEL column with single class prediction, and a column for each class from class_names, with a 0/1 for actual class presence
    #     """
    #     df = self.predictions
    #     actual_classes = self.y_test

    #     for class_name in class_names:
    #         df[class_name] = 0

    #     # Fill each row's  column classes
    #     for index, row in df.iterrows():
    #         classes = actual_classes.iloc[index][TARGET_LABEL]
    #         orig_labels = convert_str_to_list(classes)

    #         label = find_label(orig_labels, self.class_levels, self.test_level)

    #         for class_name in orig_labels:
    #             if class_name in list(df):
    #                 df.at[index, class_name] = 1

    #             undef_label = UNDEF_CLASS + label
    #             if undef_label in list(df):
    #                 df.at[index, undef_label] = 1

    #     return df

    # def get_mc_class_performance(self, class_names):
    #     """
    #     Record class performance by metrics that will later be used to compute precision and recall. Record: true positives, false positives, and # of Actual Positives, per class
    #     """
    #     df = self.combine_mc_pred_actual(class_names)
    #     class_metrics = {}
    #     # print("Unique predicted classes")
    #     # print(list(df[PRED_LABEL].unique()))
    #     for class_name in class_names:
    #         # print("for class " + class_name)

    #         AP = df.loc[df[class_name] == 1].shape[0]
    #         TP = df[(df[PRED_LABEL] == class_name) & (df[class_name] == 1)].shape[0]
    #         FP = df[(df[PRED_LABEL] == class_name) & (df[class_name] == 0)].shape[0]
    #         TN = df[(df[PRED_LABEL] != class_name) & (df[class_name] == 0)].shape[0]

    #         class_metrics[class_name] = [AP, TP, FP]

    #     return class_metrics

    def get_mc_class_accuracies(self, class_names):
        """
        Get accuracy per class : TP + TN / (TP + TN + FP + FN)
        """
        # self.predictions is DataFrame with PRED_LABEL column of class name with
        # max probability
        predicted_classes = self.predictions
        # self.y_test has TARGET_LABEL column with string list of classes per sample
        actual_classes = self.y_test
        class_accuracies = {}
        for class_name in class_names:
            class_accuracies[class_name] = 0
            for index, row in predicted_classes.iterrows():
                predicted_class = predicted_classes.iloc[index][PRED_LABEL]
                actual_classes = convert_str_to_list(
                    actual_classes.iloc[index][TARGET_LABEL])
                print("predicted class")
                print(predicted_class)
                print("actual class")
                print(actual_classes)
                if predicted_class in actual_classes:
                    class_accuracies[class_name] += 1

            class_accuracies[class_name] = class_accuracies[
                class_name] / predicted_classes.shape[0]

    def get_mc_metrics(self):
        """
        Gets recall and precision of all classes in training set (using self.get_mc_unique_classes())
        :return: {class code : recall}, {class code : precision}
        """
        class_recalls = {}
        class_precisions = {}
        for class_index, class_name in enumerate(self.class_labels):
            class_code = cat_code[class_name]
            TP, FP, FN = self.get_mc_class_recall(class_index)
            # Recall = TP/(TP+FN)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            # Precision = TP/(TP+FP)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            class_recalls[class_code] = recall
            class_precisions[class_code] = precision

        return class_recalls, class_precisions

    def get_mc_class_recall(self, class_index):
        """
        Records total number of true positives (TP), false positives (FP), and false negatives (FN) for this class across all y_test data
        :return: [TP, FP, FN]
        """
        y_test_vectors = convert_class_vectors(
            self.y_test, self.class_labels, self.class_levels)
        row_count = y_test_vectors.shape[0]
        TP = 0  # True positive count
        FN = 0  # False negative count
        FP = 0  # False positive count
        for sample_index in range(row_count - 1):

            # Compare this index of 2 class vectors
            predicted_class = self.predictions[sample_index, class_index]
            actual_classes = y_test_vectors.iloc[sample_index][TARGET_LABEL]
            actual_class = actual_classes[class_index]

            if actual_class == 1 and predicted_class == 1:
                TP += 1
            elif actual_class == 0 and predicted_class == 0:
                FN += 1
            elif actual_class == 0 and predicted_class == 1:
                FP += 1
        return [TP, FP, FN]

    def aggregate_mc_metrics(self, metrics):
        """
        Aggregate output of get_mc_metrics_by_ranges over several folds/runs
        :param metrics: Dictionary of class names to their ranged metrics,
         {class_name: [[percent_ranges, AP_ranges, TOTAL_ranges],...] , ...}
        :return: Dictionary of class names to their SUMMED ranged metrics,
         {class_name: [percent_ranges, sum(AP_ranges), sum(TOTAL_ranges)] , ...}
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

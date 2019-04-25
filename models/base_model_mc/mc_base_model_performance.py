import pandas as pd

# from models.base_model.roc_logic import *
from models.base_model.base_model_performance import BaseModelPerformance
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
        y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)
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

    # def get_all_class_rocs(self):
    #     """
    #     Gets ROC rates (TP, FP) for each sample in self.X_test/self.y_test
    #     :return class_rates: Dictionary from class_name to [fpr, tpr]
    #     """
    #     class_probabilities = self.test_probabilities()
    #     # y_test_vectors has TARGET_LABEL column, with each class vector of length
    #     # self.class_labels
    #     class_rates = {}
    #     y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)
    #     for class_index, class_name in enumerate(self.class_labels):
    #         # If there is a valid model for this class
    #         get_class_roc
    #         if models[class_name] is not None:
    #             column = class_probabilities[:, class_index]
    #             y_test_labels = self.relabel(class_index, y_test_vectors)

    #             fpr, tpr, thresholds = roc_curve(
    #                 y_true=y_test_labels, y_score=column, sample_weight=None, drop_intermediate=True)
    #             class_rates[class_name] = [fpr, tpr]
    #     return class_rates

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

    def get_mc_metrics(self):
        """
        Gets recall and precision of all classes in training set (using self.get_mc_unique_classes())
        :return: {class code : recall}, {class code : precision}
        """
        class_recalls = {}
        class_precisions = {}
        unique_classes = self.get_mc_unique_classes()
        for class_index, class_name in enumerate(self.class_labels):
            if class_name in unique_classes:
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
        y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)
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

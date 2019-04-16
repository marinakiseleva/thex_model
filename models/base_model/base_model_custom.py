import itertools
import collections
import numpy as np
import pandas as pd

from thex_data.data_consts import TARGET_LABEL, cat_code, code_cat

PRED_LABEL = 'predicted_class'

FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 300


class BaseModelCustom:
    """
    Mixin Class for BaseModel performance regarding very specific ideas
    """

    def get_rarest(self):
        """
        Gets types of samples with smallest X% probabilities assigned to type Ia.

        """
        class_name = 'Ia'
        class_code = cat_code['Ia']
        # Get probability of each sample being a Ia
        X_preds = pd.concat(
            [self.get_probability_matrix(class_code=class_code), self.test_model()], axis=1)

        def get_dist_plot(X):
            smallest_probs_X = X_preds.nsmallest(X, 'probability')

            unique_classes = self.get_unique_classes(self.y_test)
            class_counts = {class_code: 0 for class_code in unique_classes}

            for index, row in smallest_probs_X.iterrows():
                class_counts[row['actual_class']] += 1
            # Normalize counts, divide each by total
            sum_counts = sum(class_counts.values())
            class_counts = {k: v / sum_counts if sum_counts >
                            0 else 0 for k, v in class_counts.items()}

            plot_title = "Class Distribution of " + \
                str(X) + " samples with lowest probabilities for " + class_name
            self.plot_accuracies(class_counts, plot_title,
                                 class_counts=None, ylabel="Amount")
            return smallest_probs_X
        # Get X smallest probability
        smallest_probs_X = get_dist_plot(50)

        # Convert actual_class  predicted_class  to names
        smallest_probs_X['actual_class_name'] = smallest_probs_X[
            'actual_class'].apply(lambda x: code_cat[int(x)])
        smallest_probs_X['predicted_class_name'] = smallest_probs_X[
            'predicted_class'].apply(lambda x: code_cat[int(x)])

        print("\nSamples with Lowest Probability of Ia")
        print(smallest_probs_X[['actual_class_name',
                                'predicted_class_name', 'probability']])

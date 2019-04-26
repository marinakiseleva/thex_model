import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thex_data.data_consts import TARGET_LABEL, cat_code, code_cat, FIG_WIDTH, FIG_HEIGHT, DPI


class BaseModelCustom:
    """
    Mixin Class for BaseModel performance regarding very specific ideas
    """

    def plot_samples(self, index):
        X_row = self.X_test.iloc[[index]]
        y_row = self.y_test.iloc[[index]]
        self.plot_sample_distribution(X_row, y_row)

    def plot_sample_distribution(self, X_row, y_row):
        """
        Plots probability across transient types for sample
        :param X: DataFrame of features with 1 row, for single sample
        :param y: Corresponding label in DataFrame with TARGET_LABEL column
        """

        unique_classes = self.get_unique_classes()

        probabilities = self.get_class_probabilities(X_row.iloc[0])
        probabilities_list = list(probabilities.values())

        actual_class_name = code_cat[int(y_row[TARGET_LABEL])]

        # Plot classes vs probabilities
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        x_indices = np.arange(len(unique_classes))
        barlist = ax.bar(x_indices, probabilities_list, color='blue')

        class_names = [code_cat[c] for c in probabilities.keys()]
        plt.xticks(x_indices, class_names, fontsize=12)

        # Get index of bar with class_name == actual_class
        for i in range(len(class_names)):
            if class_names[i] == actual_class_name:
                # Color correct bar in green
                barlist[i].set_color('green')
                break

        plt.title('Probability Distribution for Example Galaxy')
        plt.ylabel('Probability')
        plt.xlabel('Transient Type')
        plt.show()

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

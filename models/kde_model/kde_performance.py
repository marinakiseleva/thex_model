import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

# from models.base_model.base_model_performance import *

from thex_data.data_consts import code_cat, TARGET_LABEL, ROOT_DIR


class KDEPerformance:

    def __init__(self, model):
        self.model = model

    def get_split_probabilities(self, X_test, y_test, class_code):
        """
        Get probability assigned to the actual class per row and return probabilities for positive examples (pos_probs) and negative examples (neg_probs)
        """
        unique_classes = list(set(y_test))
        class_prob_sums = self.get_class_prob_sums(self.X_test.copy(), unique_classes)
        for index, row in X_test.iterrows():
            probabilities = self.calculate_class_probabilities(row)
            prob = probabilities[class_code]

            # Probability of this class for this row
            X_test.loc[index, 'probability'] = prob

            # Whether or not this data point IS this class
            actual_class = y_test.iloc[index]
            X_test.loc[index, 'is_class'] = True if (
                actual_class == class_code) else False

        pos_probs = X_test.loc[X_test.is_class == True]['probability']
        neg_probs = X_test.loc[X_test.is_class == False]['probability']

        return pos_probs, neg_probs

    def get_class_prob_sums(self, test_X, classes):
        """
        Get normalized probabilities and predictions -- normalizing over all predictions per class per sample
        """
        # Initialize dict from class to list of probabilities
        test_X = test_X.sample(frac=1)
        class_probs = {}
        class_prob_sums = {}
        class_counts = {}
        for c in classes:
            class_probs[c] = []
            class_prob_sums[c] = 0
            class_counts[c] = 0

        for index, row in test_X.iterrows():
            probabilities = self.calculate_class_probabilities(row)
            for c in classes:
                # if class_counts[c] < 20:
                class_probs[c].append(probabilities[c])
                class_counts[c] += 1

        for c in class_probs.keys():
            class_prob_sums[c] = sum(class_probs[c])
        # print(class_prob_sums)
        return class_prob_sums

    def normalize_probabilities(self, probabilities, class_prob_sums):
        for c in probabilities.keys():
            probabilities[c] = probabilities[c] / class_prob_sums[c]
        # Normalize all probabilities to sum to 1
        for c in probabilities.keys():
            probabilities[c] = probabilities[c] / sum(probabilities.values())
        return probabilities

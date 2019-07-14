from models.base_model_mc.mc_base_model import MCBaseModel
from models.ensemble_models.mc_kde_model.mc_kde_train import MCKDETrain

import numpy as np
import pandas as pd
from thex_data.data_consts import PRED_LABEL


class MCKDEModel(MCBaseModel, MCKDETrain):
    """
    Multiclass Kernel Density Estimate (KDE) model. Creates KDE for each class, and computes probability by normalizing over class at same level in class hierarchy.
    """

    def __init__(self, **data_args):
        self.name = "Multiclass KDE Model"
        # do not use default label transformations; instead we will do it manually
        # in this class
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.models = {}

    def train_model(self):
        if self.class_labels is None:
            self.set_class_labels(self.y_train)
        print("Classes:\n------------------\n")
        print(self.class_labels)
        return self.train()

    def get_all_class_probabilities(self):
        """
        Get class probability for each sample.
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        class_densities = {}
        norm = np.zeros(self.X_test.shape[0])
        for class_index, class_name in enumerate(self.class_labels):
            kde = self.models[class_name]  # Model
            class_densities[class_name] = np.exp(
                kde.score_samples(self.X_test.values))
            norm = np.add(norm, class_densities[class_name])

        # Divide each density by normalization (sum of all densities)
        probabilities = np.zeros((self.X_test.shape[0], 0))
        for class_name in class_densities.keys():
            p_nans = np.divide(class_densities[class_name], norm)
            p = np.nan_to_num(p_nans)
            probs = np.array([p]).T  # append probabilities as column
            probabilities = np.append(probabilities, probs, axis=1)

        return probabilities

    def get_class_probabilities(self, x, normalized=True):
        """
        Calculates probability of each transient class for the single test data point (x). 
        :param x: Single row of features 
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            model = self.models[class_name]
            probabilities[class_name] = np.exp(model.score_samples([x.values]))[0]

        if normalized:
            sum_densities = sum(probabilities.values())
            probabilities = {k: v / sum_densities for k, v in probabilities.items()}
        return probabilities

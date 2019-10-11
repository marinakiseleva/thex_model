from models.ensemble_models.mc_kde_model.kde_classifier import KDEClassifier
from models.ensemble_models.ensemble_model.ensemble_model import EnsembleModel


import numpy as np


class MCKDEModel(EnsembleModel):
    """
    Ensemble Kernel Density Estimate (KDE) model. Creates KDE for each class, and computes probability by normalizing over sum of density of class and density of no class.
    """

    def __init__(self, **data_args):
        self.name = "Ensemble KDE Model"
        # do not use default label transformations - done manually
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.models = {}

    def create_classifier(self, pos_class, X, y):
        """
        Initialize classifier, with positive class as positive class name
        :param pos_class: class_name that corresponds to TARGET_LABEL == 1
        :param X: DataFrame of features
        :param y: DataFrame with TARGET_LABEL column, 1 if it has class, 0 otherwise
        """
        return KDEClassifier(pos_class, X, y)

    def get_all_class_probabilities(self, normalized=True):
        """
        Overwrite get_all_class_probabilities in EnsembleModel.
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        all_probs = np.empty((0, len(self.class_labels)))
        for index, row in self.X_test.iterrows():
            row_p = self.get_class_probabilities(row, normalized=normalized)
            all_probs = np.append(all_probs, [list(row_p.values())], axis=0)

        return all_probs

    def get_class_probabilities(self, x, normalized=True):
        """
        Calculates probability of each transient class for the single test data point (x). Probability of class 1 = density(1) / (density(1) + density(0)). 
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            pos_kde = self.models[class_name].pos_model
            neg_kde = self.models[class_name].neg_model
            pos_density = np.exp(pos_kde.score_samples([x.values]))[0]
            neg_density = np.exp(neg_kde.score_samples([x.values]))[0]
            # Normalize as binary probability first
            probabilities[class_name] = pos_density / (pos_density + neg_density)

        if normalized:
            probabilities = self.normalize_probabilities(probabilities)
        return probabilities

    def normalize_probabilities(self, probabilities):
        """
        Normalize over disjoint sets of class hierarchy.
        :param probabilities: Dictionary from class names to likelihoods 
        """
        for level in self.level_classes.keys():
            cur_level_classes = self.level_classes[level]
            # Normalize over this set of columns in probabilities
            level_sum = 0
            num_classes = 0
            for c in probabilities.keys():
                if c in cur_level_classes:
                    num_classes += 1
                    level_sum += probabilities[c]

            # Normalize by dividing each over sum
            for c in probabilities.keys():
                # If there is only 1 class in set, do not normalize
                if c in cur_level_classes and num_classes > 1:
                    probabilities[c] = probabilities[c] / level_sum
                # Threshold min probability to 0.01 to avoid out of bounds
                if probabilities[c] < 0.001:
                    probabilities[c] = 0.001
        return probabilities

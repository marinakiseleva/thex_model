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

    def get_all_class_probabilities(self):
        """
        Overwrite get_all_class_probabilities in EnsembleModel. Probability of class 1 = density(1) / (density(1) + density(0))
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """

        probabilities = np.zeros((self.X_test.shape[0], 0))
        pos_class_densities = {}
        neg_class_densities = {}
        norm = np.zeros(self.X_test.shape[0])
        for class_index, class_name in enumerate(self.class_labels):
            pos_kde = self.models[class_name].pos_model
            neg_kde = self.models[class_name].neg_model
            pos_class_densities[class_name] = np.exp(
                pos_kde.score_samples(self.X_test.values))
            neg_class_densities[class_name] = np.exp(
                neg_kde.score_samples(self.X_test.values))
            norm = np.add(neg_class_densities[class_name],
                          pos_class_densities[class_name])
            # Divide each density by (density class 1 + density class 0)
            p_nans = np.divide(pos_class_densities[class_name], norm)
            p = np.nan_to_num(p_nans)
            probs = np.array([p]).T  # append probabilities as column
            probabilities = np.append(probabilities, probs, axis=1)

        # Normalize - divide probability of each class by sum of Probabilities
        norm_probabilities = probabilities/probabilities.sum(axis=1)[:,None]
        return probabilities

    def get_class_probabilities(self, x, normalized=True):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Single row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            pos_kde = self.models[class_name].pos_model
            neg_kde = self.models[class_name].neg_model
            pos_density = np.exp(pos_kde.score_samples([x.values]))[0]
            neg_density = np.exp(neg_kde.score_samples([x.values]))[0]
            probabilities[class_name] = pos_density / (pos_density + neg_density)

        if normalized:
            total = sum(probabilities.values())
            probabilities = {k: v / total for k, v in probabilities.items()}

        return probabilities

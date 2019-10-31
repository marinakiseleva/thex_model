from models.ensemble_models.mc_kde_model.kde_classifier import KDEClassifier
from models.ensemble_models.ensemble_model.ensemble_model import EnsembleModel

from thex_data.data_consts import TREE_ROOT

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

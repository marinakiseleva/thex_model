from models.base_model.base_model import BaseModel
from models.kde_model.kde_train import KDEModelTrain
from models.kde_model.kde_test import KDEModelTest
from models.kde_model.kde_performance import KDEPerformance


class KDEModel(BaseModel, KDEModelTrain, KDEModelTest):
    """
    Model that classifies using unique Kernel Density Estimates for distributions of each feature, of each class. 
    """

    def __init__(self, cols=None, col_match=None, folds=None, **data_args):
        self.name = "KDE Model"
        self.naive = data_args['naive'] if 'naive' in data_args else False
        self.run_model(cols, col_match, folds, **data_args)

    def train_model(self):
        self.summaries, self.priors = self.train()

    def test_model(self):
        predicted_classes = self.test()
        kdep = KDEPerformance(self)
        kdep.plot_probability_metrics()
        return predicted_classes

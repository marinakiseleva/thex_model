from models.base_model.base_model import BaseModel
from models.nb_model.nb_train import train_nb
from models.nb_model.nb_test import test_nb
from models.nb_model.nb_performance import get_rocs


class NaiveBayesModel(BaseModel):
    """
    Naive Bayes Model, based on kernel density estimation of feature distributions per class.
    """

    def __init__(self, cols=None, col_match=None, folds=None, **data_args):
        self.name = "KDE Model"
        self.run_model(cols, col_match, folds, **data_args)

    def train_model(self):
        self.summaries, self.priors = train_nb(self.X_train, self.y_train)

    def test_model(self):
        predicted_classes = test_nb(self.X_test, self.summaries, self.priors)
        # get_rocs(self.X_test, self.y_test, self.summaries, self.priors)
        return predicted_classes

from models.ensemble_models.ensemble_model.ensemble_model import EnsembleModel
from models.ensemble_models.ktrees_model.tree_classifier import TreeClassifier

from thex_data.data_consts import TARGET_LABEL


class KTreesModel(EnsembleModel):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. For example, an item could have 90% probability of I and 99% of Ia.
    """

    def __init__(self, **data_args):
        self.name = "K-Trees Model"
        # do not use default label transformations; instead we will do it manually
        # in this class; model will predict multiple classes per sample
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.models = {}

    def create_classifier(self, pos_class, X, y):
        """
        Create Decision Tree classifier for pos_class versus all
        """
        return TreeClassifier(pos_class, X, y)

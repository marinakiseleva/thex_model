from models.conditional_models.conditional_ktrees_model.subktree import SubKTree
from models.conditional_models.conditional_model.conditional_model import ConditionalModel


class ConditionalKTreesModel(ConditionalModel):
    """
    Neural Networks for each group of siblings in the class hierarchy. Conditional probabilities are computed for each class.
    """

    def __init__(self, **data_args):
        super(ConditionalKTreesModel, self).__init__(**data_args)
        self.name = "Conditional K-Trees Model"

    def create_classifier(self, classes, X, y):
        return SubKTree(classes, X, y)

from models.conditional_models.conditional_dtree_model.subdtree import SubMCDTree
from models.conditional_models.conditional_model.conditional_model import ConditionalModel


class ConditionalMCDTreesModel(ConditionalModel):
    """
    Multiclass DecisionTreeClassifier for each group of siblings in the class hierarchy. Conditional probabilities are computed for each class.
    """

    def __init__(self, **data_args):
        super(ConditionalMCDTreesModel, self).__init__(**data_args)
        self.name = "Conditional Multi-Class Decision Trees Model"

    def create_classifier(self, classes, X, y):
        return SubMCDTree(classes, X, y)

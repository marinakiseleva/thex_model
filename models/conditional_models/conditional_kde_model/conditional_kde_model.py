from models.conditional_models.conditional_kde_model.subkde import SubKDE
from models.conditional_models.conditional_model.conditional_model import ConditionalModel


class ConditionalKDEModel(ConditionalModel):
    """
    KDE for each class, for each group of siblings in the class hierarchy. Normalize each probability by dividing over the sum of all sibling densities. Conditional probabilities are computed for each class.
    """

    def __init__(self, **data_args):
        super(ConditionalKDEModel, self).__init__(**data_args)
        self.name = "Conditional KDE Model"

    def create_classifier(self, classes, X, y):
        return SubKDE(classes, X, y)

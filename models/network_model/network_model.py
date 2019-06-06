from models.network_model.subnetwork import SubNetwork
from models.conditional_model.conditional_model import ConditionalModel


class NetworkModel(ConditionalModel):
    """
    Neural Networks for each group of siblings in the class hierarchy. Conditional probabilities are computed for each class.
    """

    def __init__(self, **data_args):
        super(NetworkModel, self).__init__(**data_args)
        self.name = "Neural Network Model"

    def create_classifier(self, classes, X, y):
        return SubNetwork(classes, X, y)

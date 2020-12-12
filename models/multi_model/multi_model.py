"""
Multiclass Model

Creates single multiclass classifier that compares all classes at once. Assumes all classes are independent and are not related by any class hierarchy. 

"""

from mainmodel.mainmodel import MainModel
from classifiers.multi.optmulti import OptimalMultiClassifier


class MultiModel(MainModel):

    def __init__(self, **data_args):
        """
        Initialize Multiclass Model - call init method of parent class
        """
        self.name = "Multiclass Classifier"
        super(MultiModel, self).__init__(**data_args)
        self.training_lls = []

    def train_model(self, X_train, y_train):
        """
        Train model using training data - single multiclass classifier
        """
        self.model = OptimalMultiClassifier(X=X_train,
                                            y=y_train,
                                            class_labels=self.class_labels,
                                            class_priors=self.class_priors,
                                            nb=self.nb,
                                            model_dir=self.dir)

        self.training_lls.append(self.model.training_lls)

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        probabilities = self.model.clf.get_class_probabilities(x, self.normalize)

        return probabilities

from models.base_model_mc.mc_base_model import MCBaseModel
from models.ktrees_model.ktrees_train import KTreesTrain
from models.ktrees_model.ktrees_test import KTreesTest

from thex_data.data_consts import cat_code


class KTreesModel(MCBaseModel, KTreesTrain, KTreesTest):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia.
    """

    def __init__(self, **data_args):
        self.name = "K-Trees Model"
        # do not use default label transformations; instead we will do it manually
        # in this class; model will predict multiple classes per sample
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.models = {}

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.class_labels = self.get_mc_unique_classes(self.y_train)
        return self.train()

    def test_model(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        return self.test()

    def get_all_class_probabilities(self):
        return self.test_probabilities()

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x). 
        :param x: Single row of features 
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.models[class_name]
            if tree is not None:
                class_probabilities = tree.predict_proba([x.values])
                # class_probabilities = [[prob class 0, prob class 1]]
                class_probability = class_probabilities[0][1]
            else:
                class_probability = 0

            probabilities[class_name] = class_probability

        return probabilities

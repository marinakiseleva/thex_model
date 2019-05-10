from models.base_model_mc.mc_base_model import MCBaseModel
from models.mc_kde_model.mc_kde_train import MCKDETrain
from models.mc_kde_model.mc_kde_test import MCKDETest

from thex_data.data_consts import cat_code


class MCKDEModel(MCBaseModel, MCKDETrain, MCKDETest):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia. 
    """

    def __init__(self, cols=None, col_matches=None, **data_args):
        self.name = "Multiclass KDE Model"
        # do not use default label transformations; instead we will do it manually
        # in this class
        data_args['transform_labels'] = False
        # model will predict multiple classes per sample
        self.cols = cols
        self.col_matches = col_matches
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
        :return: map from class_code to probabilities
        """
        probabilities = {}
        num_classes = len(self.models)
        for class_index, class_name in enumerate(self.class_labels):
            kde_pos = self.models[class_name][0]
            kde_neg = self.models[class_name][1]
            pos_prob_density = kde_pos.score_samples([x.values])
            neg_prob_density = kde_neg.score_samples([x.values])
            class_probability = pos_prob_density / \
                (pos_prob_density + neg_prob_density)

            probabilities[class_name] = class_probability

        return probabilities

from models.base_model_mc.mc_base_model import MCBaseModel
from models.mc_kde_model.mc_kde_train import MCKDETrain
from models.mc_kde_model.mc_kde_test import MCKDETest

from thex_data.data_consts import cat_code
from thex_data.data_consts import class_to_subclass as hierarchy
from thex_data.data_clean import *


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

        for parent in hierarchy.keys():
            hierarchy[parent].append("Undefined_" + parent)
        self.tree = init_tree(hierarchy)
        self.class_levels = assign_levels(self.tree, {}, self.tree.root, 1)

        self.test_level = 3

    def set_class_labels(self, y):
        """
        Overwrites set_class_labels. Gets all class labels on self.test_level, and undefined on this level
        """
        self.class_labels = []
        for class_name in self.get_mc_unique_classes(y):
            class_level = self.class_levels[class_name]
            if class_level == self.test_level:
                self.class_labels.append(class_name)
            elif class_level == self.test_level - 1:
                self.class_labels.append("Undefined_" + class_name)
        print(self.class_labels)

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.set_class_labels(self.y_train)
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

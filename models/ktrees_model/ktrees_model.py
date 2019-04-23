from models.base_model.base_model import BaseModel
from models.ktrees_model.ktrees_train import KTreesTrain
from models.ktrees_model.ktrees_test import KTreesTest

from thex_data.data_consts import cat_code


class KTreesModel(BaseModel, KTreesTrain, KTreesTest):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia. 
    """

    def __init__(self, cols=None, col_matches=None, **data_args):
        self.name = "K-Trees Model"
        # do not use default label transformations; instead we will do it manually
        # in this class
        data_args['transform_labels'] = False
        # model will predict multiple classes per sample
        self.cols = cols
        self.col_matches = col_matches
        self.user_data_filters = data_args

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        return self.train()

    def test_model(self):
        """
        Get class prediction for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the prediction for that class
        """
        return self.test()

    def evaluate_model(self, test_on_train):
        class_recalls, class_precisions = self.get_mc_metrics()
        self.plot_performance(class_recalls, "K-Trees Recall",
                              class_counts=None, ylabel="Recall")
        self.plot_performance(class_precisions, "K-Trees Precision",
                              class_counts=None, ylabel="Precision")

        unique_classes = self.get_mc_unique_classes()
        # Plot ROC curve
        class_rocs = {cat_code[class_name]: [] for class_name in unique_classes}
        for class_name in unique_classes:
            # Record ROC performance
            FP_rates, TP_rates = self.get_mc_roc_curve(class_name)
            class_rocs[cat_code[class_name]] = [FP_rates, TP_rates]
        self.plot_roc_curves(class_rocs, "ROC Curves")

        # Plot probability vs precision for each class
        X_accs = self.get_mc_probability_matrix()
        for class_index, class_name in enumerate(unique_classes):
            perc_ranges, AP, TOTAL = self.get_mc_metrics_by_ranges(
                X_accs, class_name)
            acc = [AP[index] / T if T > 0 else 0 for index, T in enumerate(TOTAL)]
            self.plot_probability_ranges(perc_ranges, acc,
                                         'TP/Total', class_name, TOTAL)

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x). 
        :param x: Single row of features 
        :return: map from class_code to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.ktrees[class_name]
            if tree is not None:
                class_probabilities = tree.predict_proba([x.values])
                # class_probabilities = [[prob class 0, prob class 1]]
                class_probability = class_probabilities[0][1]
            else:
                class_probability = 0

            probabilities[class_name] = class_probability

        return probabilities

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

    def train(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        # Create classifier for each class, present or not in sample
        valid_classes = []
        for class_index, class_name in enumerate(self.class_labels):
            # Relabel for this tree
            y_relabeled = self.get_class_data(class_name, self.y_train)
            positive_count = y_relabeled.loc[y_relabeled[TARGET_LABEL] == 1].shape[0]
            if positive_count < 3:
                print("No model for " + class_name)
                continue

            print("\nK-Trees Class Model: " + class_name)

            self.models[class_name] = self.create_classifier(
                class_name, self.X_train, y_relabeled)
            valid_classes.append(class_name)

        # Update class labels to only have classes for which we built models
        self.class_labels = valid_classes
        return self.models

    def create_classifier(self, pos_class, X, y):
        """
        Create Decision Tree classifier for pos_class versus all
        """
        return TreeClassifier(pos_class, X, y)

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Single row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.models[class_name].model
            if tree is not None:
                class_probabilities = tree.predict_proba([x.values])
                # class_probabilities = [[prob class 0, prob class 1]]
                class_probability = class_probabilities[0][1]
            else:
                class_probability = 0

            probabilities[class_name] = class_probability

        return probabilities

import numpy as np

from models.base_model_mc.mc_base_model import MCBaseModel
from models.ktrees_model.ktrees_train import KTreesTrain

from thex_data.data_consts import TARGET_LABEL


class KTreesModel(MCBaseModel, KTreesTrain):
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

    def get_all_class_probabilities(self):
        """
        Get class probability for each sample, from each Tree. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        # record independent probability per class
        num_samples = self.X_test.shape[0]
        default_response = np.array([[0] * num_samples]).T
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            tree = self.models[class_name]
            pos_predictions = tree.predict_proba(self.X_test)[:, 1]
            col_predictions = np.array([pos_predictions]).T
            # Add probabilities of positive as column to predictions across classes
            m_predictions = np.append(m_predictions, col_predictions, axis=1)
        return m_predictions

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

    def get_mc_class_metrics(self):
        """
        Overriding get_mc_class_metrics
        Save TP, FN, FP, TN, and BS(Brier Score) for each class.
        Brier score: (1 / N) * sum(probability - actual) ^ 2
        Log loss: -1 / N * sum((actual * log(prob)) + (1 - actual)(log(1 - prob)))
        self.y_test has TARGET_LABEL column with string list of classes per sample
        """
        # Numpy Matrix with each row corresponding to sample, and each column the
        # probability of that class
        class_accuracies = {}
        class_probabilities = self.get_all_class_probabilities()
        for class_index, class_name in enumerate(self.class_labels):
            TP = 0  # True Positives
            FN = 0  # False Negatives
            FP = 0  # False Positives
            TN = 0  # True Negatives
            BS = 0  # Brier Score
            LL = 0  # Log Loss
            class_accuracies[class_name] = 0

            # preds: probability of each sample for this class_name
            preds = class_probabilities[:, class_index]
            for index, row in self.y_test.iterrows():
                # Actual is class_name
                if class_name in row[TARGET_LABEL]:
                    # Prediction is class name (prob > 50%)
                    if preds[index] > 0.5:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if preds[index] > 0.5:
                        FP += 1
                    else:
                        TN += 1

            class_accuracies[class_name] = {"TP": TP,
                                            "FN": FN,
                                            "FP": FP,
                                            "TN": TN,
                                            "BS": BS,
                                            "LL": LL}

        return class_accuracies

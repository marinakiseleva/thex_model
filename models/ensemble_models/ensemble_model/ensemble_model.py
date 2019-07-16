from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from models.base_model_mc.mc_base_model import MCBaseModel
from thex_data.data_clean import convert_str_to_list
from thex_data.data_consts import TARGET_LABEL


class EnsembleModel(MCBaseModel, ABC):
    """
    Model that consists of K classifiers where is K is total number of all unique class labels. Each classifier compares a single class to remaining classes. Probabilities are reported unnormalized (just one versus all probabilities).
    """

    @abstractmethod
    def create_classifier(self, pos_class, X, y):
        """
        Initialize classifier, with positive class as positive class name
        """
        pass

    def train_model(self):
        """
        Train K-models, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.class_labels = self.get_mc_unique_classes(self.y_train)

        # Create classifier for each class
        valid_classes = []
        for class_index, class_name in enumerate(self.class_labels):
            y_relabeled = self.get_class_data(class_name, self.y_train)
            positive_count = y_relabeled.loc[y_relabeled[TARGET_LABEL] == 1].shape[0]
            if positive_count < 3:
                print("WARNING: No model for " + class_name)
                continue

            print("\nClass Model: " + class_name)
            self.models[class_name] = self.create_classifier(
                class_name, self.X_train, y_relabeled)
            valid_classes.append(class_name)

        # Update class labels to only have classes for which we built models
        if len(valid_classes) != len(self.class_labels):
            print("\nWARNING: Not all class labels have classifiers.")
            self.class_labels = valid_classes

        return self.models

    def get_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels. 
        :param class_name: Positive class 
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def get_all_class_probabilities(self):
        """
        Get class probability for each sample, from each model. Reconstruct class vectors for samples, with 1 if class was predicted and 0 otherwise.
        :return m_predictions: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        # record independent probability per class
        num_samples = self.X_test.shape[0]
        default_response = np.array([[0] * num_samples]).T
        m_predictions = np.zeros((num_samples, 0))
        for class_index, class_name in enumerate(self.class_labels):
            model = self.models[class_name]
            col_predictions = model.predict(self.X_test)
            # Add probabilities of positive class as column to predictions across classes
            m_predictions = np.append(m_predictions, col_predictions, axis=1)
        return m_predictions

    def get_mc_class_metrics(self):
        """
        Overriding get_mc_class_metrics in MCBaseModel to function for one versus all classification.
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

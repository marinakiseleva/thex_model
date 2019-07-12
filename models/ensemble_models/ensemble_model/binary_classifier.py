from abc import ABC, abstractmethod
from sklearn.utils.class_weight import compute_class_weight
from thex_data.data_consts import TARGET_LABEL


class BinaryClassifier(ABC):
    """
    Abstract class for binary classifier.
    """

    def __init__(self, pos_class, X, y):
        self.pos_class = pos_class
        self.classifier = self.init_classifier(X, y)

    def get_class_weights(self, labeled_samples):
        """
        Get weight of each class
        :param labeled_samples: DataFrame of features and TARGET_LABEL, where TARGET_LABEL values are 0 or 1
        :return: dictionary with 0 and 1 as keys and values of class weight
        """
        class_weights = compute_class_weight(
            class_weight='balanced', classes=[0, 1], y=labeled_samples[TARGET_LABEL].values)
        return dict(enumerate(class_weights))

    def get_sample_weights(self, labeled_samples):
        """
        Get weight of each sample (1/# of samples in class) and save in list with same order as labeled_samples
        :param labeled_samples: DataFrame of features and TARGET_LABEL, where TARGET_LABEL values are 0 or 1
        """
        classes = labeled_samples[TARGET_LABEL].unique()
        label_counts = {}
        for c in classes:
            label_counts[c] = labeled_samples.loc[
                labeled_samples[TARGET_LABEL] == c].shape[0]
        sample_weights = []
        for df_index, row in labeled_samples.iterrows():
            class_count = label_counts[row[TARGET_LABEL]]
            sample_weights.append(1 / class_count)
        # return np.array(sample_weights)
        return sample_weights

    @abstractmethod
    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Get the probability of the positive class for each row in DataFrame X. Return probabilities as Numpy column.
        :param x: 2D Numpy array as column with probability of class per row
        """
        pass

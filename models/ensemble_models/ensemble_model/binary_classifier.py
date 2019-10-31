from abc import ABC, abstractmethod
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from thex_data.data_consts import TARGET_LABEL


class BinaryClassifier(ABC):
    """
    Abstract class for binary classifier.
    """

    def __init__(self, pos_class, X, y):
        """
        Initialize binary classifier
        :param pos_class: class_name that corresponds to TARGET_LABEL == 1
        :param X: DataFrame of features
        :param y: DataFrame with TARGET_LABEL column, 1 if it has class, 0 otherwise
        """
        self.pos_class = pos_class
        self.classifier = self.init_classifier(X, y)

    def get_class_weights(self, labeled_samples):
        """
        Get weight of each class
        :param labeled_samples: DataFrame of features and TARGET_LABEL, where TARGET_LABEL values are 0 or 1
        :return: dictionary with 0 and 1 as keys and values of class weight
        """
        # Weight of class 𝑐 is the size of largest class divided by the size of class 𝑐.
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
        return sample_weights

    def get_model_dist(self, samples, p):
        """
        Returns Gaussian distribution of probabilities for the model for this set of samples.
        :param samples: DataFrame of relevant samples
        :param p:  Name of this set of samples (positiv or negative)
        """
        probabilities = []
        for index, x in samples.iterrows():
            probabilities.append(self.get_class_probability(x))
        return self.get_normal_dist(np.array(probabilities), p)

    def get_normal_dist(self, x, a):
        """
        Return Gaussian distribution, fitted with all of x
        :param x: Numpy array of values
        """
        dist = norm(loc=np.mean(x), scale=np.var(x))
        # Plot distribution of probabilities
        fig, ax = plt.subplots(1, 1)
        dist_x = np.linspace(.01, 1, 100)
        ax.plot(dist_x, dist.pdf(dist_x), 'r-',
                lw=5, alpha=0.6, label="pdf")
        ax.hist(x, density=True, histtype='stepfilled', alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("pdf(x)")
        ax.set_title("PDF with mean=" + str(np.round(np.mean(x), 2)) +
                     ", var=" + str(np.round(np.var(x), 2)))
        replace_strs = ["\n", " ", ":", ".", ",", "/"]
        pc = self.pos_class
        for r in replace_strs:
            pc = pc.replace(r, "_")

        plt.savefig("../output/dists/normaldist_" + pc + "_" + str(a))
        return dist

    @abstractmethod
    def get_class_probability(self, x):
        """
        Get probability of the positive class for this sample x
        :param x: Single row of features
        """
        pass

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

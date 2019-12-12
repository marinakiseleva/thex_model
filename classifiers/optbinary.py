"""
Construct optimal binary classifier for the data passed in 

For right now: Just optimal KDE
"""
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT
from classifiers.kdeclassifier import KDEClassifier
from classifiers.dtclassifier import DTClassifier

from sklearn.metrics import average_precision_score, brier_score_loss


class OptimalBinaryClassifier():

    def __init__(self, pos_class, X, y, priors=None):
        """
        Initialize binary classifier
        :param pos_class: class_name that corresponds to TARGET_LABEL == 1
        :param X: DataFrame of features
        :param y: DataFrame with TARGET_LABEL column, 1 if it has class, 0 otherwise
        :param priors: Priors to be used CURRENTLY NOT IMPLEMETNED
        """
        self.pos_class = pos_class
        self.opt_classifier, self.classifier_name = self.get_best_classifier(X, y)

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

    def get_class_probability(self, x):
        """
        Get class probability from optimal classifier (can only use this function after having selected best classifier)
        :param x: Pandas Series (row of DF)
        """
        return self.opt_classifier.get_class_probability(x)

    def get_class_probabilities(self, clf, X):
        """
        Get probabilities for all samples in X
        :param X: Pandas DataFrame features
        """
        probabilities = []
        for index, row in X.iterrows():
            probabilities.append(clf.get_class_probability(row))
        return np.array(probabilities)

    def get_clf_loss(self, clf, X, y):
        """
        Evaluate classifier, return average_precision_score
        :param X: Pandas DataFrame features
        :param y: Pandas DataFrame labels
        """
        predictions = self.get_class_probabilities(clf, X)
        loss = brier_score_loss(y.values.flatten(), predictions)
        return loss

    def get_best_classifier(self, X, y):
        """
        Get best classifier 
        """
        labeled_samples = pd.concat([X, y], axis=1)
        sample_weights = self.get_sample_weights(labeled_samples)
        class_weights = self.get_class_weights(labeled_samples)

        classifiers = [KDEClassifier(X, y),
                       DTClassifier(X, y, sample_weights, class_weights)]
        min_loss = 100000
        best_clf = None
        for clf in classifiers:
            tlpd = self.get_clf_loss(clf, X, y)
            if tlpd < min_loss:
                min_loss = tlpd
                best_clf = clf

        print("best clf " + str(best_clf.name))
        print('with score ' + str(min_loss))

        return best_clf, best_clf.name

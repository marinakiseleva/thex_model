"""
Construct optimal binary classifier for the data passed in. Compares performance of:

Kernel Density Estimate (KDE) classifier
Decision Tree classifier
SVM Classifier
Gaussian Naive Bayes classifier

for the given class, and keeps the best-performing one. Determines best parameters for each classifier using 3-fold cross validation.

"""
import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

from thex_data.data_consts import TARGET_LABEL
from classifiers.binary.kdeclassifier import KDEClassifier
from classifiers.binary.dtclassifier import DTClassifier
from classifiers.binary.svmclassifier import SVMClassifier
from classifiers.binary.adaboostclassifier import ADAClassifier
from classifiers.binary.gaussiannb import GNBClassifier
from classifiers.binary.kdeclassifiernb import KDENBClassifier


class OptimalBinaryClassifier():

    def __init__(self, pos_class, X, y, nb, model_dir):
        """
        Initialize binary classifier
        :param pos_class: class_name that corresponds to TARGET_LABEL == 1
        :param X: DataFrame of features
        :param y: DataFrame with TARGET_LABEL column, 1 if it has class, 0 otherwise
        :param nb: Naive Bayes boolean
        :param model_dir: Model directory
        """
        self.nb = nb
        self.dir = model_dir
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

    def get_sample_weights(self, y):
        """
        Get weight of each sample (1/# of samples in class) and save in list with same order as labeled_samples
        :param y: TARGET_LABEL, where TARGET_LABEL values are 0 or 1
        """
        return compute_sample_weight(class_weight='balanced', y=y)

    def get_class_probability(self, x, normalize=True):
        """
        Get class probability from optimal classifier (can only use this function after having selected best classifier)
        :param x: Pandas Series (row of DF)
        """
        return self.opt_classifier.get_class_probability(x, normalize)

    def get_class_probabilities(self, clf, X, normalize=True):
        """
        Get probabilities for all samples in X
        :param X: Pandas DataFrame features
        """
        probabilities = []
        for index, row in X.iterrows():
            probabilities.append(clf.get_class_probability(row, normalize))
        return np.array(probabilities)

    def get_clf_loss(self, clf, X, y):
        """
        Evaluate classifier, return brier_score_loss
        :param X: Pandas DataFrame features
        :param y: Pandas DataFrame labels
        """
        sample_weights = self.get_sample_weights(y.values)
        predictions = self.get_class_probabilities(clf, X)
        loss = brier_score_loss(y.values.flatten(), predictions,
                                sample_weight=sample_weights)
        return loss

    def train_classifiers(self, X, y, sample_weights, class_weights):
        """
        Train variety of classifiers
        """
        if self.nb:
            classifiers = [KDENBClassifier(X, y, self.pos_class, self.dir)]
        else:
            classifiers = [KDEClassifier(X, y, self.pos_class, self.dir)]
        # classifiers = [KDENBClassifier(X, y, self.pos_class, self.dir),
        #                KDEClassifier(X, y, self.pos_class, self.dir)]
        #                DTClassifier(X, y, sample_weights, class_weights),
        #                SVMClassifier(X, y, sample_weights, class_weights),
        #                GNBClassifier(X, y, sample_weights)]
        return classifiers

    def get_best_classifier(self, X, y):
        """
        Get best classifier
        """
        labeled_samples = pd.concat([X, y], axis=1)
        sample_weights = self.get_sample_weights(labeled_samples)
        class_weights = self.get_class_weights(labeled_samples)

        classifiers = self.train_classifiers(X, y, sample_weights, class_weights)

        min_loss = 100000
        best_clf = None
        for clf in classifiers:
            loss = self.get_clf_loss(clf, X, y)
            print(clf.name + " brier score loss " + str(round(loss, 3)))
            if loss < min_loss:
                min_loss = loss
                best_clf = clf

        # print("\n\nBest Classifier " + str(best_clf.name) + ' with score ' + str(min_loss))

        return best_clf, best_clf.name

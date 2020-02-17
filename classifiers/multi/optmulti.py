"""
Construct optimal multi-class classifier for the data passed in. Determines best multi-class classifier from the following:

Kernel Density Estimate (KDE) classifier
Decision Tree classifier
SVM Classifier
Gaussian Naive Bayes classifier

Determines best parameters for each classifier using 3-fold cross validation. Unlike OptimalBinaryClassifier, this does not function off an ensemble of binary classifiers, but directly compares classes. 


"""
import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import average_precision_score, brier_score_loss

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT
from classifiers.multi.multikdeclassifier import MultiKDEClassifier
from classifiers.multi.multinbkde import MultiNBKDEClassifier


class OptimalMultiClassifier():

    def __init__(self, X, y, class_labels, nb):
        """
        Multiclasss classifier. Return map from each class name in class_labels to model, and classifier name corresponding to underlying algorithm.
        :param X: DataFrame of features
        :param y: DataFrame with TARGET_LABEL column, 1 if it has class, 0 otherwise
        :param priors: Priors to be used CURRENTLY NOT IMPLEMETNED
        """
        self.class_labels = class_labels
        self.nb = nb
        self.clf, self.classifier_name = self.get_best_classifier(X, y)

    def get_one_hot(self, y):
        """
        Convert list of labels to one-hot encodings in order of self.class_labels; output 2D numpy array where each row is one-hot encoding for original label
        """

        num_labels = len(self.class_labels)
        onehot_rows = []
        for index, row in y.iterrows():
            onehot = [0 for _ in range(num_labels)]
            for index, class_label in enumerate(self.class_labels):
                if class_label in row[TARGET_LABEL]:
                    onehot[index] = 1
            onehot_rows.append(onehot)

        output = np.array(onehot_rows)

        return output

    def get_clf_loss(self, clf, X, y):
        """
        Evaluate classifier, return average_precision_score
        :param X: Pandas DataFrame features
        :param y: Pandas DataFrame labels
        """
        y_onehot = self.get_one_hot(y)
        probabilities = []
        for index, row in X.iterrows():
            row_probs = clf.get_class_probabilities(row)
            probabilities.append(np.array(list(row_probs.values())))

        loss = self.brier_multi(y_onehot, np.array(probabilities))
        return loss

    def brier_multi(self, targets, probs):
        """
        Brier score loss for multiple classes:
        https://www.wikiwand.com/en/Brier_score#/Original_definition_by_Brier
        :param targets: 2D numpy array of one hot vectors
        :param probs: 2D numpy array of probabilities, same order of classes as targets
        """
        return np.mean(np.sum((probs - targets)**2, axis=1))

    def train_classifiers(self, X, y):
        """
        Train variety of classifiers
        """
        if self.nb:
            print("\n\nTraining Naive Bayes Multiclass Classifier")
            multikde = MultiNBKDEClassifier(X, y, self.class_labels)
        else:
            print("\n\nTraining Multivariate KDE per class")
            multikde = MultiKDEClassifier(X, y, self.class_labels)
        return [multikde]

    def get_best_classifier(self, X, y):
        """
        Get best classifier 
        """
        labeled_samples = pd.concat([X, y], axis=1)
        # sample_weights = self.get_sample_weights(labeled_samples)
        # class_weights = self.get_class_weights(labeled_samples)

        classifiers = self.train_classifiers(X, y)

        min_loss = 100000
        best_clf = None
        for clf in classifiers:
            loss = self.get_clf_loss(clf, X, y)
            if loss < min_loss:
                min_loss = loss
                best_clf = clf

        print("\nBest Classifier " + str(best_clf.name) + ' with score ' + str(min_loss))

        return best_clf, best_clf.name

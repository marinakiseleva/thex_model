import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


def get_sample_weights(y):
    """
    Get weight of each sample 
    :param y: TARGET_LABEL, where TARGET_LABEL values are 0 or 1
    """
    return compute_sample_weight(class_weight='balanced', y=y)


def get_one_hot(y, class_labels):
    """
    Convert list of labels to one-hot encodings in order of self.class_labels; output 2D numpy array where each row is one-hot encoding for original label
    """
    num_labels = len(class_labels)
    onehot_rows = []
    for index, row in y.iterrows():
        onehot = [0 for _ in range(num_labels)]
        for index, class_label in enumerate(class_labels):
            if class_label in row[TARGET_LABEL]:
                onehot[index] = 1
        onehot_rows.append(onehot)

    output = np.array(onehot_rows)

    return output


def get_class_data(class_name, y):
    """
    Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
    :param class_name: Positive class
    :return: y, relabeled
    """
    labels = []  # Relabeled y
    for df_index, row in y.iterrows():
        cur_classes = thex_utils.convert_str_to_list(row[TARGET_LABEL])
        label = 1 if class_name in cur_classes else 0
        labels.append(label)
    relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
    return relabeled_y


def brier_multi(targets, probs, sample_weights):
    return np.average(np.sum((probs - targets)**2, axis=1), weights=sample_weights)


def get_class_probabilities(x, class_labels, clfs):
    """
    Get probability of each class for this sample x. Probability of class i  = density_i / (sum_i^N density_i). 
    :param x: Pandas Series (row of DF) of features
    """
    density_sum = 0
    probabilities = {}
    for class_name in class_labels:
        class_density = np.exp(clfs[class_name].score_samples([x.values]))[0]
        probabilities[class_name] = class_density
        density_sum += class_density

    # Normalize
    probabilities = {k: probabilities[k] / density_sum for k in probabilities.keys()}

    MIN_PROB = 0.0001  # Min probability to avoid overflow
    for class_name in class_labels:
        if np.isnan(probabilities[class_name]):
            probabilities[class_name] = MIN_PROB
        if probabilities[class_name] < MIN_PROB:
            probabilities[class_name] = MIN_PROB

    return probabilities


def fit_folds(X, y, bandwidth, class_labels):
    """
    Fit data using this bandwidth and kernel and report back brier score loss average over 3 folds
    """
    losses = []
    sample_weights = get_sample_weights(y)
    y_onehot = get_one_hot(y, class_labels)
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index].reset_index(
            drop=True), X.iloc[test_index].reset_index(drop=True)
        y_train, y_test = y.iloc[train_index].reset_index(
            drop=True), y.iloc[test_index].reset_index(drop=True)

        mc_kdes = {}
        for class_name in class_labels:
            mc_kdes[class_name] = KernelDensity(bandwidth=bandwidth,
                                                metric='euclidean',
                                                kernel='exponential',
                                                leaf_size=40)
            y_relabeled = get_class_data(class_name, y)
            mc_kdes[class_name].fit(X.loc[y_relabeled[TARGET_LABEL] == 1])

        probs = []  # 2D List of probabilities
        for index, x in X_test.iterrows():
            p = get_class_probabilities(x, class_labels, mc_kdes)
            probs.append(list(p.values()))
        probs = np.array(probs)
        # Evaluate using Brier Score Loss
        weights = np.take(sample_weights, test_index)
        labels = np.take(y_onehot, test_index, axis=0)
        loss = brier_multi(labels, probs, weights)
        losses.append(loss)
    # Average loss for this bandwidth across 3 folds
    avg_loss = sum(losses) / len(losses)
    return avg_loss


def find_best_bw(X, y, bandwidths, class_labels):
    """
    Fit all bandwidths for this kernel
    """
    best_cv_loss = 1000
    best_cv_bw = None
    for bandwidth in bandwidths:
        loss = fit_folds(X, y, bandwidth, class_labels)
        # Reset best BW overall.
        if loss < best_cv_loss:
            best_cv_loss = loss
            best_cv_bw = bandwidth

    # record[kernel] = [best_cv_loss, best_cv_bw]

    return best_cv_loss, best_cv_bw


class MultiKDEClassifier():
    """
    Multiclass Multivariate KDE classifier
    """

    def __init__(self, X, y, class_labels):
        """
        Init classifier through training
        """
        self.name = "Multiclass Multivariate KDE"
        self.class_labels = class_labels
        self.clfs = self.train_together(X, y)

    def train_together(self, X, y):
        """
        Train positive and negative class together using same bandwidth to try and minimize brier score loss instead of maximizing log likelihood of fit
        :param X: DataFrame of training data features
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        """
        leaf_size = 40
        bandwidths = np.linspace(0.0001, 1, 100)
        # grid = {'bandwidth': np.linspace(0, 1, 100)}
        # kernels = ['exponential', 'gaussian', 'tophat',
        #            'epanechnikov', 'cosine', 'linear']

        # best_kernel = None
        best_cv_loss, best_cv_bw = find_best_bw(X, y, bandwidths, self.class_labels)

        # best_cv_loss = 1000
        # best_cv_bw = None
        # Multiprocess by kernels
        # manager = multiprocessing.Manager()
        # record = manager.dict()
        # jobs = []

        #     cur_proc = multiprocessing.Process(
        #         target=fit_kernel,
        #         args=(X, y, leaf_size, bandwidths, kernel, self.class_labels, record))
        #     jobs.append(cur_proc)
        #     cur_proc.start()

        # Wait for all jobs to finish
        # for job in jobs:
        #     job.join()

        # Find min loss among all kernels (which is min among bandwidths)
        # for kernel_name in record.keys():
        #     best_kernel_loss, bw = record[kernel_name]
        #     if best_kernel_loss < best_cv_loss:
        #         best_cv_loss = best_kernel_loss
        #         best_cv_bw = bw
        #         best_kernel = kernel_name

        # Define models based on best bandwidth
        best_kernel = 'exponential'
        print("Best bandwidth " + str(best_cv_bw))
        print("Best kernel " + str(best_kernel))
        print("With score: " + str(best_cv_loss))
        mc_kdes = {}

        # Make KDE for each class using same bandwidth, leaf size, and kernel
        for class_name in self.class_labels:
            mc_kdes[class_name] = KernelDensity(bandwidth=best_cv_bw,
                                                leaf_size=40,
                                                kernel=best_kernel,
                                                metric='euclidean')
            y_relabeled = self.get_class_data(class_name, y)
            mc_kdes[class_name].fit(X.loc[y_relabeled[TARGET_LABEL] == 1])

        return mc_kdes

    def get_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = thex_utils.convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def train(self, X, y):
        mc_kdes = {}
        for class_name in self.class_labels:
            print("\n\nTraining: " + class_name)
            y_relabeled = self.get_class_data(class_name, y)
            X_pos = X.loc[y_relabeled[TARGET_LABEL] == 1]
            mc_kdes[class_name] = self.fit_class(X_pos)

        return mc_kdes

    def fit_class(self, X):
        """
        Fit KDE to X & select bandwidth by maximizing log-likelihood 
        :return: best fitting KDE
        """
        # Create grid to get optimal bandwidth
        grid = {'bandwidth': np.linspace(0, 1, 100)}
        num_cross_folds = 3  # number of folds in a (Stratified)KFold
        kde = KernelDensity(leaf_size=40,
                            metric='euclidean',
                            kernel='exponential')
        clf_optimize = GridSearchCV(estimator=kde,
                                    param_grid=grid,
                                    cv=num_cross_folds,
                                    iid=True,
                                    n_jobs=CPU_COUNT
                                    )
        clf_optimize.fit(X)
        clf = clf_optimize.best_estimator_

        print("Optimal KDE Parameters: " + str(clf_optimize.best_params_) +
              " \nwith log probability density (log-likelihood): " + str(clf.score(X)))

        return clf

    def get_class_probabilities(self, x, normalize=True):
        """
        Get probability of each class for this sample x. Probability of class i  = density_i / (sum_i^N density_i). 
        :param x: Pandas Series (row of DF) of features
        """
        density_sum = 0
        probabilities = {}
        for class_name in self.class_labels:
            class_density = np.exp(self.clfs[class_name].score_samples([x.values]))[0]
            probabilities[class_name] = class_density
            density_sum += class_density

        # Normalize
        if normalize:
            probabilities = {k: probabilities[k] /
                             density_sum for k in probabilities.keys()}

        MIN_PROB = 10**-15  # Min probability to avoid overflow
        for class_name in self.class_labels:
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = MIN_PROB
            if probabilities[class_name] < MIN_PROB:
                probabilities[class_name] = MIN_PROB

        return probabilities

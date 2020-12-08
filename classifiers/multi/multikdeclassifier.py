from functools import partial
import multiprocessing
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, CPU_COUNT, DEFAULT_KERNEL

###########
#  Helper functions for 'train_together' ; fitting same bandwidth to all classes


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


def fit_folds(X, y, class_labels, bandwidth):
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
                                                kernel=DEFAULT_KERNEL,
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
    print("For bandwidth " + str(bandwidth) + " loss: " + str(avg_loss))
    return avg_loss

###########

###########
# Find bandwidth using parallel processing


def get_params_ll(X, bandwidth_kernel):
    """
    Fit data using this bandwidth and kernel and report back log-likelihood fit on validation data (30% of training). This works better than 3-fold cross-validation, which was found to overfit data.
    :param X: data for only positive class
    """

    bandwidth = bandwidth_kernel[0]
    kernel = bandwidth_kernel[1]
    ll_per_fold = []  # log likelihood for each fold

    # randomly split X into training and validation
    X_train = X.sample(frac=0.7)
    X_validate = X.drop(X_train.index)
    X_train = X_train.reset_index(drop=True)
    X_validate = X_validate.reset_index(drop=True)

    kde = KernelDensity(bandwidth=bandwidth,
                        metric='euclidean',
                        kernel=kernel
                        )
    kde.fit(X_train)
    ll = kde.score(X_validate)
    return ll


def find_best_params(X, bandwidths, kernels):
    """
    Find best kernel/bandwidth pair, based on log likelihood fit, in parallel.
    """
    bw_k_pairs = []  # bandwidth -kernel pairs
    for b in bandwidths:
        for k in kernels:
            bw_k_pairs.append([b, k])
    pool = multiprocessing.Pool(CPU_COUNT)

    # Pass in parameters that don't change for parallel processes
    func = partial(get_params_ll, X)

    lls = []
    # Multithread over bandwidths
    lls = pool.map(func, bw_k_pairs)
    pool.close()
    pool.join()

    best_bw = None
    best_kernel = None

    # Maximize the log-likelihood
    min_index = np.argmax(np.array(lls))

    best_ll = lls[min_index]
    best_bw = bw_k_pairs[min_index][0]
    best_kernel = bw_k_pairs[min_index][1]

    return best_ll, best_bw, best_kernel
###########


class MultiKDEClassifier():
    """
    Multiclass Multivariate KDE classifier
    """

    def __init__(self, X, y, class_labels, class_priors):
        """
        Init classifier through training
        """
        self.name = "Multiclass Multivariate KDE"
        self.class_labels = class_labels
        self.class_priors = class_priors
        self.clfs = self.train(X, y)

    def train_together(self, X, y):
        """
        Train positive and negative class together using same bandwidth to try and minimize brier score loss instead of maximizing log likelihood of fit
        :param X: DataFrame of training data features
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        """
        leaf_size = 40
        bandwidths = np.linspace(0.001, 4, 100)

        print("Running " + str(CPU_COUNT) + " processes.")
        pool = multiprocessing.Pool(CPU_COUNT)

        # Pass in parameters that don't change for parallel processes
        func = partial(fit_folds, X, y, self.class_labels)

        # Multithread over bandwidths
        losses = []
        losses = pool.map(func, bandwidths)

        pool.close()
        pool.join()
        print("Done processing...")

        min_loss_index = losses.index(min(losses))
        best_cv_loss = losses[min_loss_index]
        best_cv_bw = bandwidths[min_loss_index]
        print("Best bandwidth " + str(best_cv_bw))
        print("With score: " + str(best_cv_loss))

        # Define models based on best bandwidth
        mc_kdes = {}
        # Make KDE for each class using same bandwidth, leaf size, and kernel
        for class_name in self.class_labels:
            mc_kdes[class_name] = KernelDensity(bandwidth=best_cv_bw,
                                                leaf_size=40,
                                                kernel=DEFAULT_KERNEL,
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
        self.training_lls = {}
        for class_name in self.class_labels:
            print("\nTraining KDE for " + class_name)
            y_relabeled = self.get_class_data(class_name, y)
            X_pos = X.loc[y_relabeled[TARGET_LABEL] == 1]
            mc_kdes[class_name] = self.fit_class(X_pos)
            self.training_lls[class_name] = self.best_kernel_ll

        return mc_kdes

    def fit_class(self, X):
        """
        Fit KDE to X & select kernel/bandwidth by maximizing log-likelihood (minimizing negative of log-likelihod)
        :return: best fitting KDE
        """
        bandwidths = np.linspace(0.0001, 1, 50)
        kernels = ['exponential']
        # 'tophat',  'epanechnikov', 'linear', 'cosine']

        best_ll, best_bw, best_kernel = find_best_params(X, bandwidths, kernels)

        kde = KernelDensity(bandwidth=best_bw,
                            metric='euclidean',
                            kernel=best_kernel)
        kde.fit(X)

        self.best_bw = best_bw
        self.best_kernel_ll = best_ll

        print("bandwidth " + str(self.best_bw))
        print("average log-likelihood: " + str(self.best_kernel_ll))

        return kde

    def get_class_probabilities(self, x, normalize=True):
        """
        Get probability of each class for this sample x. Probability of class i  = density_i / (sum_i^N density_i). 
        :param x: Pandas Series (row of DF) of features
        """
        density_sum = 0
        probabilities = OrderedDict()
        for class_index, class_name in enumerate(self.class_labels):
            class_ll = self.clfs[class_name].score_samples([x.values])[0]
            class_density = np.exp(class_ll)
            if self.class_priors is not None:
                class_density *= self.class_priors[class_index]
            probabilities[class_name] = class_density
            density_sum += class_density

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

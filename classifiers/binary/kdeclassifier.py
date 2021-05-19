import sys
import numpy as np
import multiprocessing
from functools import partial

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT, DEFAULT_KERNEL


def get_sample_weights(y):
    """
    Get weight of each sample 
    :param y: TARGET_LABEL, where TARGET_LABEL values are 0 or 1
    """
    return compute_sample_weight(class_weight='balanced', y=y)


def get_params_loss(X_train, y_train, X_validate, y_validate, bandwidth_kernel):
    """
    Fit data using this bandwidth and kernel and report back brier score loss for validation set (weighted average loss, considering class sizes.)
    """
    bandwidth = bandwidth_kernel[0]
    kernel = bandwidth_kernel[1]
    X_pos = X_train.loc[y_train[TARGET_LABEL] == 1]
    X_neg = X_train.loc[y_train[TARGET_LABEL] == 0]

    pos_model = KernelDensity(bandwidth=bandwidth,
                              metric='euclidean',
                              kernel=kernel)
    pos_model = pos_model.fit(X_pos)

    neg_model = KernelDensity(bandwidth=bandwidth,
                              metric='euclidean',
                              kernel=kernel)
    neg_model = neg_model.fit(X_neg)

    pred_probs = []
    for index, x in X_validate.iterrows():
        pos_density = pos_model.score_samples([x.values])[0]
        neg_density = neg_model.score_samples([x.values])[0]
        d = pos_density + neg_density
        if pos_density < 0 and neg_density < 0:
            pos_prob = 1 - (pos_density / d)
        elif d == 0:
            pos_prob = 0
        else:
            d = np.exp(pos_density) + np.exp(neg_density)
            if d != 0:
                pos_prob = np.exp(pos_density) / d
            else:
                pos_prob = 0
        pred_probs.append(pos_prob)

    # Evaluate using Brier Score Loss
    # Use sample weights
    sample_weights = get_sample_weights(y_validate)
    loss = brier_score_loss(y_true=y_validate.values.flatten(),
                            y_prob=pred_probs, sample_weight=sample_weights)

    return loss


def find_best_params(X, y, bandwidths, kernels):
    """
    Find best kernel/bandwidth pair, in parallel
    """
    X_train = X.sample(frac=0.7)
    y_train = y.iloc[X_train.index].reset_index(drop=True)
    X_validate = X.drop(X_train.index).reset_index(drop=True)
    y_validate = y.drop(X_train.index).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    bw_k_pairs = []  # bandwidth-kernel pairs
    for b in bandwidths:
        for k in kernels:
            bw_k_pairs.append([b, k])
    losses = []

    pool = multiprocessing.Pool(CPU_COUNT)
    # Pass in parameters that don't change for parallel processes
    func = partial(get_params_loss, X_train, y_train, X_validate, y_validate)
    # Multithread over bandwidths
    losses = pool.map(func, bw_k_pairs)
    pool.close()
    pool.join()

    # for bk in bw_k_pairs:
    #     losses.append(get_params_loss(X_train, y_train, X_validate, y_validate, bk))

    best_bw = None
    best_kernel = None

    # Minimize the loss
    min_index = np.argmin(np.array(losses))
    best_loss = losses[min_index]
    best_bw = bw_k_pairs[min_index][0]
    best_kernel = bw_k_pairs[min_index][1]

    return best_loss, best_bw, best_kernel


class KDEClassifier():
    """
    Multivariate KDE classifier
    """

    def __init__(self, X, y, pos_class, model_dir):
        """
        Init classifier through training
        :param X: DataFrame of training data features
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        :param pos_class: Name of positive class
        :param model_dir: Model directory to save output to
        """
        self.name = "Binary KDE"
        self.pos_class = pos_class

        self.train_together(X, y)

    def train_together(self, X, y):
        """
        Train positive and negative class together using same bandwidth to try and minimize brier score loss instead of maximizing log likelihood of fit
        :param X: DataFrame of training data features
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        """

        bandwidths = np.linspace(0.0001, 1, 100)
        kernels = ['exponential', 'gaussian']  # , 'gaussian', 'tophat',
        #'epanechnikov', 'linear', 'cosine']

        best_loss, best_bw, best_kernel = find_best_params(X, y, bandwidths, kernels)

        print("\n" + self.pos_class + " best bandwidth: " +
              str(best_bw) + " kernel: " + best_kernel + " with avg loss: " + str(best_loss))

        self.pos_model = KernelDensity(bandwidth=best_bw,
                                       kernel=best_kernel,
                                       metric='euclidean')
        self.pos_model.fit(X.loc[y[TARGET_LABEL] == 1])

        self.neg_model = KernelDensity(bandwidth=best_bw,
                                       kernel=best_kernel,
                                       metric='euclidean')
        self.neg_model.fit(X.loc[y[TARGET_LABEL] == 0])

    def train(self, X):
        """
        Get maximum likelihood estimated distribution by kernel density estimate of this class over all features
        :return: best fitting KDE
        """
        # Create grid to get optimal bandwidth
        grid = {'bandwidth': np.linspace(0, 1, 100)}
        num_cross_folds = 3  # number of folds in a (Stratified)KFold
        kde = KernelDensity(leaf_size=10,
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

        print(self.name + " optimal parameters:\n" + str(clf_optimize.best_params_))
        print("with log probability density (log-likelihood): " + str(clf.score(X)))
        sys.stdout.flush()  # Print to output file
        return clf

    def get_class_probability(self, x, normalize=True):
        """
        Get probability of this class for this sample x. Probability of class 1 = density(1) / (density(1) + density(0)). 
        :param x: Pandas Series (row of DF) of features
        """
        pos_density = np.exp(self.pos_model.score_samples([x.values]))[0]
        neg_density = np.exp(self.neg_model.score_samples([x.values]))[0]
        d = pos_density + neg_density
        if d == 0:
            pos_prob = 0
        else:
            if normalize:
                pos_prob = pos_density / (pos_density + neg_density)
            else:
                pos_prob = pos_density
        return pos_prob

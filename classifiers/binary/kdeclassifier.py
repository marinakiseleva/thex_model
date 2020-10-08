import sys
import numpy as np
import multiprocessing
from functools import partial

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


def get_sample_weights(y):
    """
    Get weight of each sample 
    :param y: TARGET_LABEL, where TARGET_LABEL values are 0 or 1
    """
    return compute_sample_weight(class_weight='balanced', y=y)


def get_params_loss(X, y, kernel, bandwidth):
    """
    Fit data using this bandwidth and kernel and report back brier score loss average over 3 folds, with samples weighted 
    """
    losses = []
    sample_weights = get_sample_weights(y)
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index].reset_index(
            drop=True), X.iloc[test_index].reset_index(drop=True)
        y_train, y_test = y.iloc[train_index].reset_index(
            drop=True), y.iloc[test_index].reset_index(drop=True)

        X_pos = X_train.loc[y_train[TARGET_LABEL] == 1]
        X_neg = X_train.loc[y_train[TARGET_LABEL] == 0]

        pos_model = KernelDensity(bandwidth=bandwidth,
                                  metric='euclidean',
                                  kernel=kernel
                                  )
        pos_model.fit(X_pos)
        neg_model = KernelDensity(bandwidth=bandwidth,
                                  metric='euclidean',
                                  kernel=kernel)
        neg_model.fit(X_neg)

        pos_probs = []
        for index, x in X_test.iterrows():
            pos_density = np.exp(pos_model.score_samples([x.values]))[0]
            neg_density = np.exp(neg_model.score_samples([x.values]))[0]
            d = pos_density + neg_density
            if d == 0:
                pos_prob = 0
            else:
                pos_prob = pos_density / (pos_density + neg_density)
            pos_probs.append(pos_prob)

        # Evaluate using Brier Score Loss
        weights = np.take(sample_weights, test_index)

        # origloss = brier_score_loss(y_test.values.flatten(), pos_probs)

        loss = brier_score_loss(y_true=y_test.values.flatten(),
                                y_prob=pos_probs, sample_weight=weights)

        losses.append(loss)
    # Average loss for this bandwidth across 3 folds
    avg_loss = sum(losses) / len(losses)
    return avg_loss


def fit_kernel_parallel(X, y, bandwidths, kernel):
    """
    Fit all bandwidths to kernel, and get min loss bandwidth, in parallel.
    """

    pool = multiprocessing.Pool(CPU_COUNT)

    # Pass in parameters that don't change for parallel processes
    func = partial(get_params_loss, X, y, kernel)

    losses = []
    # Multithread over bandwidths
    losses = pool.map(func, bandwidths)
    pool.close()
    pool.join()
    print("Done processing...")

    best_cv_loss = 1000
    best_cv_bw = None
    for index, bandwidth in enumerate(bandwidths):
        loss = losses[index]
        # Reset best BW overall.
        if loss < best_cv_loss:
            best_cv_loss = loss
            best_cv_bw = bandwidth

    return best_cv_loss, best_cv_bw


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
        bandwidths = np.linspace(0.0001, 3, 100)

        best_kernel = 'exponential'

        best_cv_loss, best_cv_bw = fit_kernel_parallel(X, y,
                                                       bandwidths,
                                                       kernel=best_kernel)

        print(self.pos_class + " optimal bandwidth: " +
              str(best_cv_bw) + " with loss: " + str(best_cv_loss))

        self.pos_model = KernelDensity(bandwidth=best_cv_bw,
                                       kernel=best_kernel,
                                       metric='euclidean')
        self.pos_model.fit(X.loc[y[TARGET_LABEL] == 1])

        self.neg_model = KernelDensity(bandwidth=best_cv_bw,
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

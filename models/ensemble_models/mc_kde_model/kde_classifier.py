import pandas as pd
import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from scipy.stats import norm
# import matplotlib.pyplot as plt


from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class KDEClassifier(BinaryClassifier):
    """
    Extension of abstract class for binary classifier.
    """

    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        # Fit KDE to positive samples only.
        X_pos = X.loc[y[TARGET_LABEL] == 1]
        y_pos = y.loc[y[TARGET_LABEL] == 1]
        self.pos_model = self.get_best_model(X_pos, y_pos)

        # need negative model to normalize over
        X_neg = X.loc[y[TARGET_LABEL] == 0]
        y_neg = y.loc[y[TARGET_LABEL] == 0]
        self.neg_model = self.get_best_model(X_neg, y_neg)

        self.pos_dist = self.get_model_dist(X_pos, "positive")
        self.neg_dist = self.get_model_dist(X_neg, "negative")

        return self.pos_model

    def get_model_dist(self, samples, p):
        """
        Returns Gaussian distribution of probabilities for the model. 
        :param marg_model: Refers to the model we need to marginalize densities over to get probabilities: model / model + marg_model
        """
        pos_densities = np.exp(self.pos_model.score_samples(samples))
        neg_densities = np.exp(self.neg_model.score_samples(samples))
        probs = pos_densities / (pos_densities + neg_densities)
        return self.get_normal_dist(probs, p)

    def get_normal_dist(self, x, a):
        """
        Return Gaussian distribution, fitted with 1/3 of x
        """
        dist = norm(loc=np.mean(x), scale=np.var(x))
        # Plot distribution of probabilities
        # fig, ax = plt.subplots(1, 1)
        # dist_x = np.linspace(.01, 1, 100)
        # ax.plot(dist_x, dist.pdf(dist_x), 'r-',
        #         lw=5, alpha=0.6, label='norm pdf')
        # ax.hist(x, density=True, histtype='stepfilled', alpha=0.2)
        # plt.savefig("../output/dists/normaldist_" + str(self.pos_class) + "_" + str(a))
        return dist

    def predict(self, X):
        """
        Get the probability of the positive class for each row in DataFrame X. Return probabilities as Numpy column.
        :param x: 2D Numpy array as column with probability of class per row
        """
        raise ValueError(
            "KDE Model overwrites get_all_class_probabilities so it does not need predict.")

    def get_best_model(self, X, y):
        """
        Get maximum likelihood estimated distribution by kernel density estimate of this class over all features
        :return: best fitting KDE
        """
        # Create grid to get optimal bandwidth
        range_bws = np.linspace(0.01, 5, 200)
        grid = {
            'bandwidth': range_bws,
            'kernel': ['gaussian'],
            'metric': ['euclidean']
        }
        clf_optimize = GridSearchCV(KernelDensity(), grid,
                                    iid=False, cv=3, n_jobs=CPU_COUNT)
        clf_optimize.fit(X)
        # print("Optimal Parameters:")
        # print(clf_optimize.best_params_)

        clf = clf_optimize.best_estimator_

        # print("Total log-likelihood of training data: " + str(clf.score(X)))

        return clf

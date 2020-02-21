import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import brier_score_loss

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


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
        # Fit KDE to positive samples only.
        # X_pos = X.loc[y[TARGET_LABEL] == 1]
        # self.pos_model = self.train(X_pos)

        # # need negative model to normalize over
        # X_neg = X.loc[y[TARGET_LABEL] == 0]
        # self.neg_model = self.train(X_neg)

    def train_together(self, X, y):
        """
        Train positive and negative class together using same bandwidth to try and minimize brier score loss instead of maximizing log likelihood of fit
        :param X: DataFrame of training data features
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        """
        leaf_size = 40
        bandwidths = np.linspace(0.0001, 5, 100)

        skf = StratifiedKFold(n_splits=3, shuffle=True)
        best_cv_loss = 1000
        best_cv_bw = None
        for bandwidth in bandwidths:
            bw_losses = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index].reset_index(
                    drop=True), X.iloc[test_index].reset_index(drop=True)
                y_train, y_test = y.iloc[train_index].reset_index(
                    drop=True), y.iloc[test_index].reset_index(drop=True)

                X_pos = X_train.loc[y_train[TARGET_LABEL] == 1]
                X_neg = X_train.loc[y_train[TARGET_LABEL] == 0]

                kde_pos = KernelDensity(bandwidth=bandwidth,
                                        leaf_size=leaf_size,
                                        metric='euclidean',
                                        kernel='exponential')
                kde_pos.fit(X_pos)
                kde_neg = KernelDensity(bandwidth=bandwidth,
                                        leaf_size=leaf_size,
                                        metric='euclidean',
                                        kernel='exponential')
                kde_neg.fit(X_neg)

                pos_probs = []
                for index, x in X_test.iterrows():
                    pos_density = np.exp(kde_pos.score_samples([x.values]))[0]
                    neg_density = np.exp(kde_neg.score_samples([x.values]))[0]
                    d = pos_density + neg_density
                    if d == 0:
                        pos_prob = 0
                    else:
                        pos_prob = pos_density / (pos_density + neg_density)
                    pos_probs.append(pos_prob)

                # Evaluate using Brier Score Loss
                loss = brier_score_loss(y_test.values.flatten(), pos_probs)
                bw_losses.append(loss)

            # Average loss for this bandwidth across 3 folds
            avg_loss = sum(bw_losses) / len(bw_losses)

            # Reset best BW overall.
            if avg_loss < best_cv_loss:
                best_cv_loss = avg_loss
                best_cv_bw = bandwidth

        # Define models based on best bandwidth

        self.pos_model = KernelDensity(bandwidth=best_cv_bw,
                                       leaf_size=leaf_size,
                                       metric='euclidean',
                                       kernel='exponential')
        self.pos_model.fit(X.loc[y[TARGET_LABEL] == 1])

        self.neg_model = KernelDensity(bandwidth=best_cv_bw,
                                       leaf_size=leaf_size,
                                       metric='euclidean',
                                       kernel='exponential')
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

    def get_class_probability(self, x):
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
            pos_prob = pos_density / (pos_density + neg_density)
        return pos_prob

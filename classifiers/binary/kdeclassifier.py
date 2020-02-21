import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class KDEClassifier():
    """
    KDE classifier
    """

    def __init__(self, X, y, pos_class, model_dir):
        """
        Init classifier through training
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        """
        self.name = "Binary KDE"
        self.pos_class = pos_class
        # Fit KDE to positive samples only.
        X_pos = X.loc[y[TARGET_LABEL] == 1]
        self.pos_model = self.train(X_pos)

        # need negative model to normalize over
        X_neg = X.loc[y[TARGET_LABEL] == 0]
        self.neg_model = self.train(X_neg)

    def train(self, X):
        """
        Get maximum likelihood estimated distribution by kernel density estimate of this class over all features
        :return: best fitting KDE
        """
        # Create grid to get optimal bandwidth and kernel
        grid = {'bandwidth': np.linspace(0.00001, 1, 100)}
        clf_optimize = GridSearchCV(estimator=KernelDensity(kernel='exponential',
                                                            metric='euclidean'),
                                    param_grid=grid,
                                    cv=3,  # number of folds in a (Stratified)KFold
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

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class KDEClassifier():
    """
    KDE classifier
    """

    def __init__(self, X, y):
        """
        Init classifier through training
        """
        self.name = "KDE"
        # Fit KDE to positive samples only.
        X_pos = X.loc[y[TARGET_LABEL] == 1]
        y_pos = y.loc[y[TARGET_LABEL] == 1]
        self.pos_model = self.train(X_pos)

        # need negative model to normalize over
        X_neg = X.loc[y[TARGET_LABEL] == 0]
        y_neg = y.loc[y[TARGET_LABEL] == 0]
        self.neg_model = self.train(X_neg)

    def train(self, X):
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
        clf_optimize = GridSearchCV(estimator=KernelDensity(),
                                    param_grid=grid,
                                    cv=3,  # number of folds in a (Stratified)KFold
                                    )
        clf_optimize.fit(X)
        # print("Optimal Parameters:")
        # print(clf_optimize.best_params_)

        clf = clf_optimize.best_estimator_
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

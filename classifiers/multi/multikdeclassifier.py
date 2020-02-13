import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity

import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class MultiKDEClassifier():
    """
    Multiclass KDE classifier
    """

    def __init__(self, X, y, class_labels):
        """
        Init classifier through training
        """
        self.name = "Multiclass KDE"
        self.class_labels = class_labels
        self.clfs = self.train(X, y)

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
        Fit KDE to positive class presence
        :return: best fitting KDE
        """
        # Create grid to get optimal bandwidth & kernel
        grid = {
            'bandwidth': np.linspace(0.01, 6, 1000),
            'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear'],
            'metric': ['euclidean']
        }
        clf_optimize = GridSearchCV(estimator=KernelDensity(),
                                    param_grid=grid,
                                    cv=3,  # number of folds in a (Stratified)KFold
                                    iid=True,
                                    n_jobs=CPU_COUNT
                                    )
        clf_optimize.fit(X)
        clf = clf_optimize.best_estimator_

        print("Optimal KDE Parameters: " + str(clf_optimize.best_params_) +
              " \nwith log probability density (log-likelihood): " + str(clf.score(X)))

        return clf

    def get_class_probabilities(self, x):
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
        probabilities = {k: probabilities[k] / density_sum for k in probabilities.keys()}

        for class_name in self.class_labels:
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = 0.001
                print("MultiKDEClassifier get_class_probabilities NULL probability for " + class_name)

            if probabilities[class_name] < 0.0001:
                # Force min prob to 0.001 for future computation
                probabilities[class_name] = 0.001

        return probabilities

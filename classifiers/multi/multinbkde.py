import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity

import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, CPU_COUNT
from classifiers.multi.plot_fit import plot_fit


class MultiNBKDEClassifier():
    """
    Multiclass Naive Bayes classifier, with unique KDE per class per feature 
    """

    def __init__(self, X, y, class_labels, model_dir):
        """
        Init classifier through training
        """
        self.name = "Multiclass NB KDE"
        self.class_labels = class_labels
        self.dir = model_dir
        # self.clfs is map of class name to : map from feature to best fit KDE for
        # this feature/class
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
            sys.stdout.flush()  # Print to output file
            y_relabeled = self.get_class_data(class_name, y)
            X_pos = X.loc[y_relabeled[TARGET_LABEL] == 1]
            mc_kdes[class_name] = self.fit_class(X_pos, class_name)

        return mc_kdes

    def fit_class(self, X, class_name):
        """
        Fit KDE per feature separately. If there is no data for a feature, do not make a KDE.
        :return: best fitting KDEs
        """
        features = list(X)
        # Create grid to get optimal bandwidth & kernel
        grid = {
            'bandwidth': np.linspace(0, 1, 10),
            'kernel': ['tophat',  'exponential', 'gaussian'],
        }
        num_cross_folds = 3  # number of folds in a (Stratified)KFold
        kde = KernelDensity(leaf_size=10,
                            metric='euclidean')
        clf_optimize = GridSearchCV(estimator=kde,
                                    param_grid=grid,
                                    cv=num_cross_folds,
                                    iid=True,
                                    n_jobs=CPU_COUNT
                                    )
        feature_kdes = {}
        for feature in features:
            feature_data = X[feature]
            feature_data.dropna(inplace=True)
            feature_data = feature_data.values.reshape(-1, 1)
            if np.size(feature_data) > 9:
                clf_optimize.fit(feature_data)
                clf = clf_optimize.best_estimator_
                feature_kdes[feature] = clf
                print(feature)
                print("# of samples: " + str(np.size(feature_data)))
                print("KDE params: " + str(clf_optimize.best_params_) +
                      " with log probability density (log-likelihood): " + str(clf.score(feature_data)))
                # plot_fit(data=feature_data,
                #          kde=clf,
                #          feature_name=feature,
                #          class_name=class_name,
                #          model_dir=self.dir)
            else:
                feature_kdes[feature] = None
                print(str(feature) + " has no data.")
            sys.stdout.flush()  # Print to output file

        return feature_kdes

    def get_class_density(self, x, clf):
        """
        :param x: Numpy array of features for single data point
        :param clf: Map from feature to best fit KDE for a particular class
        """

        scores = []
        for feature in clf.keys():
            # Ensure this class has a defined KDE for this feature
            if clf[feature] is not None:
                # Ensure this sample has a defined value for this feature
                x_feature = x[feature]
                if x_feature is not None and not np.isnan(x_feature):
                    density = np.exp(clf[feature].score_samples([[x_feature]])[0])

                    scores.append(density)

        if len(scores) == 0:
            scores = [0]  # Probability is 0 when no features overlap with available KDEs

        return np.prod(np.array(scores))

    def get_class_probabilities(self, x):
        """
        Get probability of each class for this sample x by normalizing over each class KDE density. Probability of class i  = density_i / (sum_i^N density_i). 
        :param x: Pandas Series (row of DF) of features
        """
        density_sum = 0
        probabilities = {}
        for class_name in self.class_labels:
            class_density = self.get_class_density(x, self.clfs[class_name])
            probabilities[class_name] = class_density
            density_sum += class_density

        # Normalize
        probabilities = {k: probabilities[k] / density_sum for k in probabilities.keys()}

        MIN_PROB = 0.0001  # Min probability to avoid overflow
        for class_name in self.class_labels:
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = MIN_PROB
                print(self.name + " NULL probability for " + class_name)

            if probabilities[class_name] < MIN_PROB:
                # Force min prob to 0.001 for future computation
                probabilities[class_name] = MIN_PROB

        return probabilities

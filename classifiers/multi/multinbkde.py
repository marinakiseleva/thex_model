import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, CPU_COUNT, DEFAULT_KERNEL
from classifiers.plot_fit import plot_fits


class MultiNBKDEClassifier():
    """
    Multiclass Naive Bayes classifier, with unique KDE per class per feature
    """

    def __init__(self, X, y, class_labels, model_dir):
        """
        Init classifier through training
        """
        self.name = "Naive Bayes Multiclass KDE"
        self.class_labels = class_labels
        self.dir = model_dir
        # Record log prob density fit of data per KDE
        self.class_feat_fits = {}
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
        self.all_features = sorted(list(X))

        for class_name in self.class_labels:
            print("\n\nTraining: " + class_name)
            sys.stdout.flush()  # Print to output file
            y_relabeled = self.get_class_data(class_name, y)

            X_pos = X.loc[y_relabeled[TARGET_LABEL] == 1]
            mc_kdes[class_name] = self.fit_class(X_pos, class_name)

        plot_fits(kdes=mc_kdes,
                  features=self.all_features,
                  classes=self.class_labels,
                  model_dir=self.dir)
        return mc_kdes

    def filter_estimates(self, kdes):
        """
        Keep only KDEs with high log likelihood fit to the training data. Drop X features with lowest log likelihoods.
        :param kdes: Map from class name to map from feature to KDE 
        """
        num_drop = 3
        for class_name in self.class_feat_fits.keys():
            min_features = []
            for feature_name in self.class_feat_fits[class_name].keys():
                score = self.class_feat_fits[class_name][feature_name]
                if len(min_features) < num_drop:
                    min_features.append(feature_name)
                else:
                    # Check if score is less than all scores of all features in current
                    # list, replace feature if it is
                    for cur_feat in min_features:
                        cur_score = self.class_feat_fits[class_name][cur_feat]
                        if score < cur_score:
                            min_features.remove(cur_feat)
                            min_features.append(feature_name)
            # Now reset min features KDEs to None
            for feature in min_features:
                kdes[class_name][feature] = None

        return kdes

    def fit_class(self, X, class_name):
        """
        Fit KDE per feature separately. If there is no data for a feature, do not make a KDE.
        :return: best fitting KDEs
        """
        # Create grid to get optimal bandwidth
        grid = {'bandwidth': np.linspace(0, 1, 100)}
        num_cross_folds = 3  # number of folds in a (Stratified)KFold
        kde = KernelDensity(leaf_size=10,
                            metric='euclidean',
                            kernel=DEFAULT_KERNEL)
        clf_optimize = GridSearchCV(estimator=kde,
                                    param_grid=grid,
                                    cv=num_cross_folds,
                                    iid=True,
                                    n_jobs=CPU_COUNT
                                    )
        self.class_feat_fits[class_name] = {}
        feature_kdes = {}
        for feature in self.all_features:
            feature_data = X[feature]
            feature_data.dropna(inplace=True)
            feature_data = feature_data.values.reshape(-1, 1)
            if np.size(feature_data) > 9:
                clf_optimize.fit(feature_data)
                clf = clf_optimize.best_estimator_
                feature_kdes[feature] = clf

                # Record and print fit of this KDE
                log_density_fit = clf.score(feature_data)
                self.class_feat_fits[class_name][feature] = log_density_fit

                print(feature)
                print("# of samples: " + str(np.size(feature_data)))
                print("KDE params: " + str(clf_optimize.best_params_))
                print("with log-likelihood: " + str(log_density_fit))

            else:
                feature_kdes[feature] = None
                print(str(feature) + " has no data.")
            sys.stdout.flush()  # Print to output file

        return feature_kdes

    def get_class_density(self, x, clf, features):
        """
        Get class probability density by multiplying probability densities of each feature together. Ensures that same number of features are being used across classes by using passed in features only.
        :param x: Numpy array of features for single data point
        :param clf: Map from feature to best fit KDE for a particular class
        """
        scores = []
        for feature in features:
            x_feature = x[feature]
            density = np.exp(clf[feature].score_samples([[x_feature]])[0])
            scores.append(density)

        # Probability is 0 when no features overlap with available KDEs
        if len(scores) == 0:
            scores = [0]

        return np.prod(np.array(scores))

    def get_features(self, x):
        """
        Get maximum list of features that have valid values for x, and valid KDEs for classes in self.class_labels
        """

        valid_features = []
        for feature in self.all_features:
            if x[feature] is not None and not np.isnan(x[feature]):
                # Ensure it is valid for each classifier, otherwise we cannot use it
                add = True
                for name, clf in self.clfs.items():
                    if clf[feature] is None:
                        add = False
                        break
                if add:
                    valid_features.append(feature)
        return valid_features

    def get_class_probabilities(self, x, normalize=True):
        """
        Get probability of each class for this sample x by normalizing over each class KDE density. Probability of class i  = density_i / (sum_i^N density_i).
        :param x: Pandas Series (row of DF) of features
        """
        density_sum = 0
        probabilities = {}
        valid_features = self.get_features(x)

        for class_name in self.class_labels:
            class_density = self.get_class_density(
                x, self.clfs[class_name], valid_features)
            probabilities[class_name] = class_density
            density_sum += class_density

        # Normalize
        if normalize:
            probabilities = {k: probabilities[k] /
                             density_sum for k in probabilities.keys()}

        MIN_PROB = 0.0001  # Min probability to avoid overflow
        for class_name in self.class_labels:
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = MIN_PROB
                print(self.name + " NULL probability for " + class_name)

            if probabilities[class_name] < MIN_PROB:
                probabilities[class_name] = MIN_PROB

        return probabilities

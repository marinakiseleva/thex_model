import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from thex_data.data_consts import TARGET_LABEL, CPU_COUNT, DEFAULT_KERNEL
from classifiers.plot_fit import plot_fits


class KDENBClassifier():
    """
    KDE Naive Bayes classifier
    """

    def __init__(self, X, y, pos_class, model_dir):
        """
        Init classifier through training
        :param X: DataFrame of training data features
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        :param pos_class: Name of positive class
        :param model_dir: Model directory to save output to
        """
        self.name = "Naive Bayes Binary KDE"

        self.all_features = sorted(list(X))

        self.pos_class = pos_class
        # Fit KDE to positive samples only.
        X_pos = X.loc[y[TARGET_LABEL] == 1]
        self.pos_model = self.train(X_pos)

        # need negative model to normalize over
        X_neg = X.loc[y[TARGET_LABEL] == 0]
        self.neg_model = self.train(X_neg)

        mc_kdes = {pos_class: self.pos_model,
                   "Not " + pos_class: self.neg_model}

        plot_fits(kdes=mc_kdes,
                  features=self.all_features,
                  classes=[pos_class, "Not " + pos_class],
                  model_dir=model_dir)

    def train(self, X):
        """
        Fit KDE per feature separately. If there is no data for a feature, do not make a KDE.
        :return: best fitting KDEs
        """
        # Create grid to get optimal bandwidth
        grid = {'bandwidth': np.linspace(0, 1, 100)}
        num_cross_folds = 3  # number of folds in a (Stratified)KFold
        kde = KernelDensity(leaf_size=40,
                            metric='euclidean',
                            kernel=DEFAULT_KERNEL)
        clf_optimize = GridSearchCV(estimator=kde,
                                    param_grid=grid,
                                    cv=num_cross_folds,
                                    iid=True,
                                    n_jobs=CPU_COUNT
                                    )
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
        Get maximum list of features that have valid values for x, and valid KDEs for classes
        """
        valid_features = []
        for feature in self.all_features:
            # Ensure feature is valid for pos and neg classifier
            if x[feature] is not None and not np.isnan(x[feature]) and self.pos_model[feature] is not None and self.neg_model[feature] is not None:
                valid_features.append(feature)
        return valid_features

    def get_class_probability(self, x):
        """
        Get probability of positive class for this sample x by normalizing over each positive + negative KDE density.
        :param x: Pandas row of features
        """
        density_sum = 0
        probabilities = {}
        valid_features = self.get_features(x)

        pos_density = self.get_class_density(x, self.pos_model, valid_features)
        neg_density = self.get_class_density(x, self.neg_model, valid_features)

        d = pos_density + neg_density
        if d == 0:
            pos_prob = 0
        else:
            pos_prob = pos_density / (pos_density + neg_density)
        return pos_prob

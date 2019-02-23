import random
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from thex_data.data_consts import ROOT_DIR, TARGET_LABEL, code_cat
from thex_data.data_print import print_priors


"""
Logic for training the KDE classifier
"""


class KDEModelTrain:
    """
    Mixin Class for KDE Model Training functionality, used in KDEModel
    """

    def train(self):
        """
        Train KDE classifier
        """
        data = pd.concat([self.X_train, self.y_train], axis=1)
        classes = list(data[TARGET_LABEL].unique())
        priors = self.set_priors(data, classes)
        class_data = self.separate_classes(data, classes)

        summaries = {}
        for class_code, instances in class_data.items():
            if self.naive:
                summaries[class_code] = self.get_naive_distributions(
                    instances, code_cat[class_code])
            else:
                # Do not assume feature independence. Create distribution over all
                # features at once
                summaries[class_code] = self.get_class_distribution(
                    instances, class_code)

        return summaries, priors

    def set_priors(self, data, unique_classes, prior_type="uniform"):
        """
        Set prior probabilities of classes (defaults to uniform)
        :param data: X_train concat with y_train
        :param classes: list of unique classes
        """
        priors = {}  # Prior value of class, based on frequency
        for class_code in unique_classes:
            class_data = data.loc[data[TARGET_LABEL] == class_code]
            if prior_type == "uniform":
                priors[class_code] = 1 / len(unique_classes)
            elif prior_type == "inv_frequency":
                priors[class_code] = 1  (class_data.shape[0] / data.shape[0])
            else:  # traditional, frequency-based
                priors[class_code] = class_data.shape[0] / data.shape[0]

        # Make priors sum to 1
        priors = {k: round(v / sum(priors.values()), 6) for k, v in priors.items()}
        return priors

    def separate_classes(self, data, unique_classes):
        """
        Split data by class. Return map of {class code : DataFrame of samples of that type}
        :param data: X_train concat with y_train
        :param classes: list of unique classes
        """
        separated_classes = {}
        for class_code in unique_classes:
            # Add data for this class to mapping
            class_data = data.loc[data[TARGET_LABEL] == class_code]
            separated_classes[class_code] = class_data.drop([TARGET_LABEL], axis=1)
        return separated_classes

    def get_class_distribution(self, X_class, class_code):
        """
        Get maximum likelihood estimated distribution by kernel density estimate of this class over all features
        :param X_class: DataFrame of data of with target=class_name
        :param class_code: Class code of X_class
        """
        # Create grid to search over bandwidth for
        range_bws = np.linspace(0.01, 2, 100)
        grid = GridSearchCV(
            KernelDensity(), {'bandwidth': range_bws}, cv=3)
        grid.fit(X_class.values)
        bw = grid.best_params_['bandwidth']
        kde = KernelDensity(bandwidth=bw, kernel='gaussian', metric='euclidean')
        return kde.fit(X_class.values)

    def get_best_bandwidth(self, data, ps):
        """
        Estimate best bandwidth value based on iteratively running over different bandwidths
        :param data: Pandas Series of particular feature
        """
        range_bws = np.linspace(0.01, 2, 100)

        # Find bandwidth between min and max that maximizes likelihood
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': range_bws}, cv=ps)
        grid.fit([[cv] for cv in data])
        return grid.best_params_['bandwidth']

    def get_feature_distribution(self, data, ps, feature, class_name):
        """
        Uses kernel density estimation to find the distribution of feature. Returns [distribution, parameters of distribution].
        :param data: values in Pandas Series of feature and class
        :param feature: Name of feature/column this data corresponds to
        :param class_name: Name of class this data corresponds to
        """

        # Use Kernel Density
        bw = self.get_best_bandwidth(data, ps)
        kde = KernelDensity(bandwidth=bw, kernel='gaussian', metric='euclidean')
        # select column of values for this feature
        kde = kde.fit(np.matrix(data.values).T)

        self.plot_dist_fit(data.values, kde, bw, feature,
                           "Kernel Distribution with bandwidth: %.6f\n for feature %s in class %s" % (bw, feature, class_name))

        return kde

    def get_naive_distributions(self, X_class, class_name=None):
        """
        Estimate distribution of each feature in X_class. Return mapping of {feature : [distribution, parameters of distribution]}
        :param X_class: DataFrame of data of with target=class_name
        :param class_name: Name of class this X_class corresponds to
        """
        class_summaries = {}

        # Set training and testing indices for CV in KDE
        # 0 for validation, -1 for training
        num_samples = X_class.shape[0]
        test_assignments = [-1] * num_samples
        validation_indices = random.sample(
            list(enumerate(test_assignments)), int(num_samples * 0.3))
        # validation_indices is randomly selected tuples of (index, value)
        for i in validation_indices:
            test_assignments[i[0]] = 0
        ps = PredefinedSplit(test_assignments)

        for feature in X_class:
            class_summaries[feature] = self.get_feature_distribution(
                X_class[feature], ps, feature, class_name)

        return class_summaries

    def plot_dist_fit(self, data, kde, bandwidth, feature, title):
        """
        Plot distribution fitted to feature
        """
        # plt.ioff()
        rcParams['figure.figsize'] = 6, 6
        n_bins = 20
        range_vals = data.max() - data.min()
        x_line = np.linspace(data.min(),
                             data.max(), 1000)
        data_vector = np.matrix(x_line).T
        pdf = kde.score_samples(data_vector)
        fig, ax1 = plt.subplots()

        ax1.hist(data, n_bins, fc='gray',  alpha=0.3)
        ax1.set_ylabel('Count', color='gray')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Kernel Density', color='blue')
        ax2.plot(x_line, np.exp(pdf), linewidth=3,
                 alpha=0.5, label='bw=%.2f' % bandwidth)
        ax1.set_xlabel(feature)
        plt.title(title)
        replace_strs = ["\n", " ", ":", ".", ",", "/"]
        for r in replace_strs:
            title = title.replace(r, "_")
        plt.savefig(ROOT_DIR + "/output/kernel_fits/" + title)
        plt.close()
        # plt.cla()

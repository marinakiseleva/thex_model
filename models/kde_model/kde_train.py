import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from thex_data.data_consts import ROOT_DIR

import scipy.stats as stats
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

from thex_data.data_consts import TARGET_LABEL, code_cat
from thex_data.data_print import print_priors


"""
Logic for training the KDE classifier
"""


class KDEModelTrain:
    """
    KDE Model Training functionality, used in KDEModel
    """

    def train(self):
        """
        Train Naive Bayes classifier on this training set
        :param X_train: Features of data
        :param y_train: Labels of data
        """
        separated, priors = self.separate_classes()
        summaries = {}
        for class_code, instances in separated.items():
            summaries[class_code] = self.summarize(instances, code_cat[class_code])
        return summaries, priors

    def get_best_bandwidth(self, data):
        """
        Estimate best bandwidth value based on iteratively running over different bandwidths
        :param data: Pandas Series of particular feature
        """
        # Determine best bandwidth of kernel
        range_vals = data.values.max() - data.values.min()
        min_val = data.values.min() - (range_vals / 5)
        max_val = data.values.max() + (range_vals / 5)
        min_bw = abs(max_val - min_val) / 200
        max_bw = abs(max_val - min_val) / 5
        iter_value = 100
        # Find bandwidth between min and max that maximizes likelihood
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(min_bw, max_bw, iter_value)}, cv=3)
        grid.fit([[cv] for cv in data])
        bw = grid.best_params_['bandwidth']
        if bw == 0:
            bw = 0.001
        return bw

    def get_best_fitting_dist(self, data, feature=None, class_name=None):
        """
        Uses kernel density estimation to find the distribution of this data. Also plots data and distribution fit to it. Returns [distribution, parameters of distribution].
        :param data: Pandas Series, set of values corresponding to feature and class
        :param feature: Name of feature/column this data corresponds to
        :param class_name: Name of class this data corresponds to
        """

        # Use Kernel Density
        bw = self.get_best_bandwidth(data)
        kde = KernelDensity(bandwidth=bw, kernel='gaussian', metric='euclidean')
        best_dist = kde.fit(np.matrix(data.values).T)
        best_params = kde.get_params()

        vals_2d = [[cv] for cv in data]
        # self.plot_dist_fit(data.values, kde, bw, feature, "Kernel Distribution with bandwidth: %.6f\n for feature %s in class %s" % (bw, feature, class_name))

        # Return best fitting distribution and parameters (loc and scale)
        return [best_dist, best_params]

    def plot_dist_fit(self, data, kde, bandwidth, feature, title):
        """
        Plot distribution fitted to feature
        """
        plt.ioff()
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
        # plt.show()
        plt.close()
        plt.cla()

    def summarize(self, data, class_name=None):
        """
        Estimate distribution of each feature in this data. Return mapping of {feature : [distribution, parameters of distribution]}
        :param data: DataFrame corresponding to all data of this class (class_name)
        :param class_name: Name of class this data corresponds to
        """
        class_summaries = {}
        # get distribution of each feature
        for feature in data:
            if feature != TARGET_LABEL:
                col_values = data[feature].dropna(axis=0)
                if len(col_values) > 0:
                    class_summaries[feature] = self.get_best_fitting_dist(
                        col_values, feature, class_name)

        return class_summaries

    def separate_classes(self):
        """
        Separate by class (of unique transient types) and assigns priors (uniform)
        Return map of {class code : DataFrame of samples of that type}, and priors
        :param data: DataFrame of feature and labels
        """
        data = pd.concat([self.X_train, self.y_train], axis=1)
        transient_classes = list(data[TARGET_LABEL].unique())
        separated_classes = {}
        priors = {}  # Prior value of class, based on frequency
        total_count = data.shape[0]
        for transient in transient_classes:
            trans_df = data.loc[data[TARGET_LABEL] == transient]

            # Priors

            # Frequency-based
            # priors[transient] = trans_df.shape[0] / total_count

            # Uniform prior
            priors[transient] = 1 / len(transient_classes)

            # Inverted Frequency-based prior
            # priors[transient] = 1 - (trans_df.shape[0] / total_count)

            # Set class value
            trans_df.drop([TARGET_LABEL], axis=1, inplace=True)
            separated_classes[transient] = trans_df

        # Make priors sum to 1
        priors = {k: round(v / sum(priors.values()), 6) for k, v in priors.items()}
        # print_priors(priors)
        return separated_classes, priors

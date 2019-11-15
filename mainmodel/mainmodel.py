"""
Base model structure which models all build off of. Model structure allows for flexibility in normalization method and hierarchy use. Model defines structure that all submodels have in common:

Initial data collection class
Evaluation strategy (k-fold cross validation)
Performance aggregation

"""
import os
import shutil
from abc import ABC, abstractmethod

from sklearn.model_selection import StratifiedKFold


from thex_data.data_init import *
from thex_data.data_prep import get_source_target_data
from thex_data.data_plot import *
from thex_data.data_consts import TARGET_LABEL
from thex_data.data_consts import class_to_subclass as class_hier

import utilities.utilities as util


class MainModel(ABC):

    def __init__(self, **user_data_filters):
        """
        Initialize model based on user arguments
        """
        self.init_file_directories()

        data_filters = {'cols': None,  # Names of columns to filter on; default is all numeric cols
                        'col_matches': None,  # String by which columns will be selected
                        'num_runs': 1,
                        'folds': 3,  # Number of folds if using k-fold Cross Validation
                        'subsample': None,
                        'transform_features': True,  # Derive mag colors
                        'incl_redshift': True,
                        'min_class_size': 9,
                        'pca': None,  # Number of principal components
                        'class_labels': None,
                        'prior': 'uniform'
                        }

        for data_filter in user_data_filters.keys():
            data_filters[data_filter] = user_data_filters[data_filter]

        # list of features to use from database
        features = collect_cols(data_filters['cols'], data_filters['col_matches'])

        X, y = get_source_target_data(features, **data_filters)

        self.class_labels = self.get_class_labels(data_filters['class_labels'], y)

        self.visualize_data(X, y)
        print("Class labels " + str(self.class_labels))

        results = self.run_cfv(X, y, data_filters['folds'], data_filters['num_runs'])

    def init_file_directories(self):
        # Create output directory
        if not os.path.exists(ROOT_DIR + "/output"):
            os.mkdir(ROOT_DIR + "/output")

        file_dir = ROOT_DIR + "/output/" + util.clean_str(self.name)
        # Clear old output directories, if they exist
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)
        os.mkdir(file_dir)

    def get_class_labels(self, user_defined_labels, y):
        """
        Get class labels over which to run analysis, either from the user defined parameter, or from the data itself. If there are no user defined labels, we compile a list of unique transient type labels based on what is known in the hierarchy.
        :param user_defined_labels: None, or valid list of class labels
        :param y: DataFrame with TARGET_LABEL column
        """
        if user_defined_labels is not None:
            return user_defined_labels

        defined_classes = []
        for k, v in class_hier.items():
            defined_classes.append(k)
            for c in v:
                defined_classes.append(k)
        defined_classes = set(defined_classes)

        # Only keep classes that exist in data
        data_labels = []
        for index, row in y.iterrows():
            labels = util.convert_str_to_list(row[TARGET_LABEL])
            for label in labels:
                data_labels.append(label)
        data_labels = set(data_labels)
        return list(data_labels.intersection(defined_classes))

    def get_class_counts(self, y):
        """
        Returns count of each class in self.class_labels in y
        :param y: Pandas DataFrame with TARGET_LABEL column
        :return class_counts: Map of {class_name : count, ...}
        """
        class_counts = {}
        for class_name in self.class_labels:
            count = 0
            for index, row in y.iterrows():
                labels = util.convert_str_to_list(row[TARGET_LABEL])
                if class_name in labels:
                    count += 1
            class_counts[class_name] = count
        return class_counts

    def visualize_data(self, X, y):
        """
        """
        class_counts = self.get_class_counts(y)
        plot_class_hist(class_counts, util.clean_str(self.name))
        # Combine X and y for plotting feature dist
        df = pd.concat([X, y], axis=1)
        features = list(df)
        if 'redshift' in features:
            plot_feature_distribution(df, util.clean_str(
                self.name), 'redshift', self.class_labels, class_counts)

    def visualize_performance(self, results):
        """
        Visualize performance
        """
        print("visualize_performance not yet implemented")
        return -1

    def run_cfv(self, X, y, k, runs):
        """
        Run k-fold cross validation over a number of runs
        :param k: Number of folds
        :param runs: Number of runs to aggregate results over
        """
        kf = StratifiedKFold(n_splits=k, shuffle=True)

        results = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index].reset_index(
                drop=True), X.iloc[test_index].reset_index(drop=True)
            y_train, y_test = y.iloc[train_index].reset_index(
                drop=True), y.iloc[test_index].reset_index(drop=True)

            self.train_model(X_train, y_train)

            # Test model
            probabilities = self.get_all_class_probabilities(X_test)
            # TODO: Save probabilities + self.y_test labels! for later evaluation.
            results.append(probabilities)
        return results

    def get_all_class_probabilities(self, X_test):
        """
        Get class probabilities for all test data.
        :return probabilities: Numpy 2D Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        all_probs = np.empty((0, len(self.class_labels)))
        for index, row in X_test.iterrows():
            row_p = self.get_class_probabilities(row)
            all_probs = np.append(all_probs, [list(row_p.values())], axis=0)

        return all_probs

    @abstractmethod
    def get_class_probabilities(self, x):
        """
        Get class probabilities for sample x
        :return: map from class_name to probabilities, with classes in order of self.class_labels
        """
        pass

    @abstractmethod
    def train_model(self, X_train, X_test):
        """
        Train model using training data
        """
        pass

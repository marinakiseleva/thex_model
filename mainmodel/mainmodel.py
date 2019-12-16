"""
Base model structure which models all build off of. Model structure allows for flexibility in normalization method and hierarchy use. Model defines structure that all submodels have in common:

Initial data collection class
Evaluation strategy (k-fold cross validation)
Performance aggregation

Subclasses differ in how they normalize across the class hierarchy.

"""

from abc import ABC, abstractmethod

from sklearn.model_selection import StratifiedKFold
import pandas as pd
from hmc import hmc

from thex_data.data_init import *
from thex_data.data_prep import get_source_target_data
from thex_data.data_filter import filter_class_size, sub_sample
from thex_data.data_plot import *
from thex_data.data_consts import TARGET_LABEL, TREE_ROOT, UNDEF_CLASS, class_to_subclass as orig_class_hier

from mainmodel.performance_plots import MainModelVisualization
import utilities.utilities as util


class MainModel(ABC, MainModelVisualization):

    def __init__(self, **user_data_filters):
        """
        Initialize model based on user arguments
        """
        util.init_file_directories(self.name)

        # Must add Unspecifieds to tree, so that when searching for lowest-level
        # label, the UNDEF one returns.
        self.class_hier = orig_class_hier
        for parent in orig_class_hier.keys():
            self.class_hier[parent].insert(0, UNDEF_CLASS + parent)

        self.tree = self.init_tree(self.class_hier)
        self.class_levels = self.assign_levels(self.tree, {}, self.tree.root, 1)

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
                        'prior': 'uniform',
                        'data': None
                        }

        for data_filter in user_data_filters.keys():
            data_filters[data_filter] = user_data_filters[data_filter]

        # list of features to use from database
        features = collect_cols(data_filters['cols'], data_filters['col_matches'])

        if data_filters['data'] is None:
            X, y = get_source_target_data(features, **data_filters)
        else:
            X = data_filters['data'][0]
            y = data_filters['data'][1]

        # Redefine labels with Unspecifieds
        y = self.add_unspecified_labels_to_data(y)

        self.class_labels = self.get_class_labels(data_filters['class_labels'], y)

        X, y = self.filter_data(X, y,
                                data_filters['min_class_size'],
                                data_filters['subsample'])

        self.X = X
        self.y = y

        self.num_folds = data_filters['folds']
        self.num_runs = data_filters['num_runs']

    def filter_data(self, X, y, min_class_size, max_class_size):
        """
        Filter data now that class labels are known.
        """
        # Filter data (keep only classes that are > min class size)
        data = pd.concat([X, y], axis=1)
        filtered_data = filter_class_size(data,
                                          min_class_size,
                                          self.class_labels)
        if max_class_size is not None:
              # Randomly subsample any over-represented classes down to passed-in value
            filtered_data = sub_sample(filtered_data,
                                       max_class_size,
                                       self.class_labels)
        X = filtered_data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
        y = filtered_data[[TARGET_LABEL]]
        return X, y

    def run_model(self):
        """
        Visualize data, run analysis, and record results.
        """
        self.visualize_data(self.X, self.y)
        print("Class labels " + str(self.class_labels))

        results = self.run_cfv(self.X, self.y, self.num_folds, self.num_runs)

        self.visualize_performance(results, self.y)

    def init_tree(self, hierarchy):
        print("\n\nConstructing Class Hierarchy Tree...")
        hmc_hierarchy = hmc.ClassHierarchy(TREE_ROOT)
        for parent in hierarchy.keys():
            # hierarchy maps parents to children, so get all children
            list_children = hierarchy[parent]
            for child in list_children:
                # Nodes are added with child parent pairs
                try:
                    hmc_hierarchy.add_node(child, parent)
                except ValueError as e:
                    print(e)
        return hmc_hierarchy

    def assign_levels(self, tree, mapping, node, level):
        """
        Assigns level to each node based on level in hierarchical tree. The lower it is in the tree, the larger the level. The level at the root is 1. 
        :return: Dict from class name to level number.
        """
        mapping[str(node)] = level
        for child in tree._get_children(node):
            self.assign_levels(tree, mapping, child, level + 1)
        return mapping

    def get_class_labels(self, user_defined_labels, y):
        """
        Get class labels over which to run analysis, either from the user defined parameter, or from the data itself. If there are no user defined labels, we compile a list of unique transient type labels based on what is known in the hierarchy.
        :param user_defined_labels: None, or valid list of class labels
        :param y: DataFrame with TARGET_LABEL column
        """
        if user_defined_labels is not None:
            return user_defined_labels

        defined_classes = set()
        for k, class_list in self.class_hier.items():
            defined_classes.add(k)
            for c in class_list:
                defined_classes.add(c)
        # Only keep classes that exist in data
        data_labels = set()
        for index, row in y.iterrows():
            labels = util.convert_str_to_list(row[TARGET_LABEL])
            for label in labels:
                data_labels.add(label)

        class_labels = list(data_labels.intersection(defined_classes))

        return list(data_labels.intersection(defined_classes))

    def add_unspecified_labels_to_data(self, y):
        """
        Add unspecified label for each tree parent in data's list of labels
        """
        for index, row in y.iterrows():
            # Iterate through all class labels for this label
            max_depth = 0  # Set max depth to determine what level is undefined
            for label in util.convert_str_to_list(row[TARGET_LABEL]):
                if label in self.class_levels:
                    max_depth = max(self.class_levels[label], max_depth)
            # Max depth will be 0 for classes unhandled in hierarchy.
            if max_depth > 0:
                # Add Undefined label for any nodes at max depth
                for label in util.convert_str_to_list(row[TARGET_LABEL]):
                    if label in self.class_levels and self.class_levels[label] == max_depth:
                        add = ", " + UNDEF_CLASS + label
                        y.iloc[index] = y.iloc[index] + add
        return y

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
        plot_class_hist(self.name, class_counts)
        # Combine X and y for plotting feature dist
        df = pd.concat([X, y], axis=1)
        features = list(df)
        if 'redshift' in features:
            plot_feature_distribution(self.name,  df, 'redshift',
                                      self.class_labels, class_counts)

    def visualize_performance(self, results, y):
        """
        Visualize performance
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """

        range_metrics = self.compute_probability_range_metrics(results)
        self.plot_probability_vs_accuracy(range_metrics)
        class_metrics = self.compute_metrics(results)
        self.plot_all_metrics(class_metrics, y)

        return -1

    def scale_data(self, X_train, y_train, X_test, y_test):
        """
        Fit scaling to training data and apply to both training and testing
        """
        return None

    def apply_PCA(self, X_train, y_train, X_test, y_test):
        """
        Fit PCA to training data and apply to both training and testing
        """
        return None

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

            # TODO: Scale and/or apply PCA here.

            self.train_model(X_train, y_train)

            # Test model
            probabilities = self.get_all_class_probabilities(X_test)
            # Add labels as column to probabilities, for later evaluation
            label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
            probabilities = np.hstack((probabilities, label_column))
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

    @abstractmethod
    def compute_probability_range_metrics(self, results):
        """
        Computes True Positive & Total metrics, split by probability assigned to class for ranges of 10% from 0 to 100. Used to plot probability assigned vs completeness.
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :return range_metrics: Map of classes to [TP_range_sums, total_range_sums]
            total_range_sums: # of samples with probability in range for this class
            TP_range_sums: true positives per range 
        """
        pass

    @abstractmethod
    def compute_metrics(self, results):
        """
        Compute TP, FP, TN, and FN per class.
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """
        pass

    @abstractmethod
    def compute_baselines(self, class_counts, y):
        """
        Get random classifier baselines for recall, specificity (negative recall), and precision
        :param prior: NEED TO REINCORPORATE.
        """
        pass

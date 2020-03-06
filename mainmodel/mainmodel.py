"""
Base model structure which models all build off of. Model structure allows for flexibility in normalization method and hierarchy use. Model defines structure that all submodels have in common:

Initial data collection class
Evaluation strategy (k-fold cross validation)
Performance aggregation

Subclasses differ in how they normalize across the class hierarchy.

"""
import sys
import pickle
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from hmc import hmc

# local imports
from thex_data.data_init import *
from thex_data.data_prep import get_source_target_data
from thex_data.data_filter import filter_class_size, sub_sample, super_sample
from thex_data.data_plot import *
from thex_data.data_consts import TARGET_LABEL, TREE_ROOT, UNDEF_CLASS, class_to_subclass as orig_class_hier
from mainmodel.performance_plots import MainModelVisualization
import utilities.utilities as util


class MainModel(ABC, MainModelVisualization):

    def __init__(self, **user_data_filters):
        """
        Initialize model based on user arguments
        """
        self.dir = util.init_file_directories(self.name)
        print("Saving " + self.name + " output to directory " + self.dir)
        # Redirect prints to log
        self.orig_stdout = sys.stdout  # Save to reset later
        sys.stdout = open(self.dir + "/experiment.log", "a")

        # Must add Unspecifieds to tree, so that when searching for lowest-level
        # label, the UNDEF one returns.
        self.class_hier = orig_class_hier.copy()
        for parent in orig_class_hier.keys():
            self.class_hier[parent].insert(0, UNDEF_CLASS + parent)
        self.tree = self.init_tree(self.class_hier)
        self.class_levels = self.assign_levels(self.tree, {}, self.tree.root, 1)

        data_filters = {'cols': None,  # Names of columns to filter on; default is all numeric cols
                        'col_matches': None,  # String by which columns will be selected
                        'num_runs': None,
                        'folds': None,  # Number of folds if using k-fold Cross Validation
                        'subsample': None,
                        'supersample': None,
                        'transform_features': True,  # Derive mag colors
                        'min_class_size': 9,
                        'pca': None,  # Number of principal components
                        'class_labels': None,
                        'prior': 'uniform',
                        'data': None,  # List of training and test pandas dfs
                        'nb': False  # Naive Bayes multiclass
                        }

        for data_filter in user_data_filters.keys():
            data_filters[data_filter] = user_data_filters[data_filter]

        if data_filters['data'] is None:
            # list of features to use from database
            features = collect_cols(data_filters['cols'], data_filters['col_matches'])
            X, y = get_source_target_data(features, data_filters)
        else:
            X = data_filters['data'][0].copy()
            y = data_filters['data'][1].copy()

        # Redefine labels with Unspecifieds
        y = self.add_unspecified_labels_to_data(y)

        self.class_labels = self.get_class_labels(
            data_filters['class_labels'], y, data_filters['min_class_size'])

        # Pre-processing dependent on class labels
        X, y = self.filter_data(X, y, data_filters, self.class_labels)

        print("\nClasses Used:\n" + str(self.class_labels))
        print("\nFeatures Used:\n" + str(list(X)))

        # Save relevant data attributes to self
        self.X = X
        self.y = y
        self.num_folds = data_filters['folds']
        self.num_runs = data_filters['num_runs']
        self.pca = data_filters['pca']
        self.oversample = data_filters['supersample']
        self.nb = data_filters['nb']

    def run_model(self):
        """
        Visualize data, run analysis, and record results.
        """
        print("\nRunning " + str(self.name))

        self.visualize_data(self.X, self.y)

        if self.num_runs is not None:
            results = self.run_trials(self.X, self.y, self.num_runs)
        elif self.num_folds is not None:
            results = self.run_cfv(self.X, self.y)
        else:
            raise ValueError("Must pass in either number of folds or runs.")

        # Save results in pickle
        with open(self.dir + '/results.pickle', 'wb') as f:
            pickle.dump(results, f)
        with open(self.dir + '/y.pickle', 'wb') as f:
            pickle.dump(self.y, f)

        self.visualize_performance(results, self.y)

        sys.stdout = self.orig_stdout

    def filter_data(self, X, y, data_filters, class_labels):
        """
        Filter data now that class labels are known.
        """
        min_class_size = data_filters['min_class_size']
        max_class_size = data_filters['subsample']

        # Filter data (keep only classes that are > min class size)
        data = pd.concat([X, y], axis=1)
        filtered_data = filter_class_size(data,
                                          min_class_size,
                                          class_labels)

        if max_class_size is not None:
              # Randomly subsample any over-represented classes down to passed-in value
            filtered_data = sub_sample(filtered_data,
                                       max_class_size,
                                       class_labels)

        X = filtered_data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
        y = filtered_data[[TARGET_LABEL]].reset_index(drop=True)

        X, y = self.drop_conflicts(X, y)
        return X, y

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

    def get_class_labels(self, user_defined_labels, y, N):
        """
        Keep all classes with # of samples > min class size. If a class has only 1 child class that meets this criteria, we only use the parent class and not the children. We ensure each class is greater than N, and then we make disjoint by keeping only leaves of the new class hierarchy tree. If user_defined_labels is not NULL, we filter on this at the end.
        :param user_defined_labels: None, or valid list of class labels
        :param y: DataFrame with TARGET_LABEL column
        :param N: Minimum # of samples per class to keep it
        """
        # Initialize new hierarchy and class counts
        new_hier = self.class_hier.copy()

        # Get counts of all defined classes
        defined_classes = set()
        for k, class_list in self.class_hier.items():
            defined_classes.add(k)
            for c in class_list:
                defined_classes.add(c)
        self.class_labels = list(defined_classes)
        counts = self.get_class_counts(y)

        # 1. if only 1 child has > N samples, just use parent
        drop_parents = []
        for parent in new_hier.keys():
            children = new_hier[parent]
            num_good_children = 0
            for child in children:
                if child in counts and counts[child] > N:
                    num_good_children += 1
            if num_good_children <= 1:
                # Use parent only, by dropping this key
                drop_parents.append(parent)
        for p in drop_parents:
            del new_hier[p]

        # 2. Make sure each child class has at least MIN samples
        for parent in new_hier.keys():
            keep_children = []
            for child in new_hier[parent]:
                if child in counts and counts[child] > N:
                    keep_children.append(child)
            new_hier[parent] = keep_children

        # 3. Keep leaves
        keep_classes = []
        for parent in new_hier.keys():
            for child in new_hier[parent]:
                if child not in new_hier.keys():
                    keep_classes.append(child)

        # 4. Filter on any classes passed in by user
        if user_defined_labels is not None:
            keep_classes = list(set(keep_classes).intersection(set(user_defined_labels)))

        self.class_hier = new_hier
        return keep_classes

    def get_class_counts(self, y):
        """
        Returns count of each class in self.class_labels in y
        :param y: Pandas DataFrame with TARGET_LABEL column
        :return class_counts: Map of {class_name : count, ...}
        """
        class_counts = {c: 0 for c in self.class_labels}
        for index, row in y.iterrows():
            for class_name in self.class_labels:
                labels = util.convert_str_to_list(row[TARGET_LABEL])
                if class_name in labels:
                    class_counts[class_name] += 1
        return class_counts

    def visualize_data(self, X, y):
        """
        Visualize data completeness and distribution
        """
        # visualize_completeness(self.dir, X, y, self.class_labels)

        class_counts = self.get_class_counts(y)
        plot_class_hist(self.dir, class_counts)
        # Combine X and y for plotting feature dist
        df = pd.concat([X, y], axis=1)
        features = list(df)
        if 'redshift' in features:
            plot_feature_distribution(self.dir,  df, 'redshift',
                                      self.class_labels, class_counts)

    def visualize_performance(self, results, y):
        """
        Visualize performance
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """
        class_counts = self.get_class_counts(y)
        range_metrics = self.compute_probability_range_metrics(results)
        self.plot_prob_pr_curves(range_metrics, class_counts)
        # self.plot_probability_vs_accuracy(range_metrics)
        class_metrics, set_totals = self.compute_metrics(results)
        self.plot_all_metrics(class_metrics, set_totals, y)

        return -1

    def scale_data(self, X_train, X_test):
        """
        Fit scaling to training data and apply to both training and testing;
        Returns X_train and X_test as Pandas DataFrames
        :param X_train: Pandas DataFrame of training data
        :param X_test: Pandas DataFrame of testing data
        """
        features_list = list(X_train)
        # Rescale data: z = (x - mean) / stdev
        scaler = StandardScaler()
        scaled_X_train = pd.DataFrame(
            data=scaler.fit_transform(X_train), columns=features_list)
        scaled_X_test = pd.DataFrame(
            data=scaler.transform(X_test), columns=features_list)

        return scaled_X_train, scaled_X_test

    def apply_PCA(self, X_train, X_test):
        """
        Fit PCA to training data and apply to both training and testing;
        Returns X_train and X_test as Pandas DataFrames
        :param X_train: Pandas DataFrame of training data
        :param X_test: Pandas DataFrame of testing data
        """
        k = self.pca

        def convert_to_df(data, k):
            """
            Convert Numpy 2D array to DataFrame with k PCA columns
            :param data: Numpy 2D array of data features
            :param k: Number of PCA components to label cols
            """
            reduced_columns = []
            for i in range(k):
                new_column = "PC" + str(i + 1)
                reduced_columns.append(new_column)
            df = pd.DataFrame(data=data, columns=reduced_columns)

            return df

        # Return original if # features <= # PCA components
        if len(list(X_train)) <= k:
            return X_train, X_test

        pca = PCA(n_components=k)
        reduced_training = pca.fit_transform(X_train)
        reduced_testing = pca.transform(X_test)
        print("\nPCA Analysis: Explained Variance Ratio")
        print(pca.explained_variance_ratio_)

        reduced_training = convert_to_df(reduced_training, k)
        reduced_testing = convert_to_df(reduced_testing, k)
        return reduced_training, reduced_testing

    def super_sample(self, X, y):
        """
        Super sample using self.oversample as number to sample up to
        """
        data = pd.concat([X, y], axis=1)
        super_data = super_sample(data, self.oversample, self.class_labels)

        X = super_data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
        y = super_data[[TARGET_LABEL]]

        return X, y

    def run_cfv(self, X, y):
        """
        Run k-fold cross validation over a number of runs
        :param X: DataFrame of features data
        :param y: DataFram with TARGET_LABEL column
        """
        kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True)
        results = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index].reset_index(
                drop=True), X.iloc[test_index].reset_index(drop=True)
            y_train, y_test = y.iloc[train_index].reset_index(
                drop=True), y.iloc[test_index].reset_index(drop=True)

            # Oversample training data only
            if self.oversample is not None:
                X_train, y_train = self.super_sample(X_train, y_train)

            # Scale and apply PCA
            X_train, X_test = self.scale_data(X_train, X_test)
            if self.pca is not None:
                X_train, X_test = self.apply_PCA(X_train, X_test)

            # Train
            self.train_model(X_train, y_train)

            # Test model
            probabilities = self.get_all_class_probabilities(X_test)
            # Add labels as column to probabilities, for later evaluation
            label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
            probabilities = np.hstack((probabilities, label_column))
            results.append(probabilities)

        return results

    def drop_conflicts(self, X, y):
        """
        Drop rows that have more than 1 disjoint label assigned (conflicting labels)
        """
        keep_indices = []
        for index, row in y.iterrows():
            keep = True
            labels = util.convert_str_to_list(row[TARGET_LABEL])
            # Row shouldn't contain more than 1 label in a set of children.
            for c_key in self.class_hier.keys():
                children = self.class_hier[c_key]
                com = list(set(labels).intersection(set(children)))
                if len(com) > 1:
                    keep = False
                    break
            if keep:
                keep_indices.append(index)

        k = list(set(keep_indices))
        return X.loc[k, :].reset_index(drop=True), y.loc[k, :].reset_index(drop=True)

    def manually_stratify(self, X, y, train_split):
        """
        Stratify data manually - meaning classes are properly distributed in both training and testing data.
        """
        c = self.get_class_counts(y)
        if y.shape[0] != sum(c.values()):
            raise ValueError(
                "Data size does not equal class counts. This indicates that something is wrong with disjoint labels.")
        # Split data by class
        class_data = {c: [] for c in self.class_labels}
        class_indices = []
        for index, row in y.iterrows():
            for class_name in self.class_labels:
                labels = util.convert_str_to_list(row[TARGET_LABEL])
                if class_name in labels:
                    class_data[class_name].append(index)

        train = []
        test = []
        # Get train/test indices per class
        for class_name in class_data.keys():
            indices = np.array(class_data[class_name])
            # Shuffle
            np.random.shuffle(indices)
            # First part training
            train += list(indices[:int(len(indices) * train_split)])
            # Second part testing
            test += list(indices[int(len(indices) * train_split):])

        # Ensure that no indices are repeated
        if len(train) != len(list(set(train))):
            raise ValueError("Sample used more than once.")

            # Filter data on train/test indices
        X_train = X.iloc[train, :].reset_index(drop=True)
        y_train = y.iloc[train, :].reset_index(drop=True)
        X_test = X.iloc[test, :].reset_index(drop=True)
        y_test = y.iloc[test, :].reset_index(drop=True)
        return X_train, y_train, X_test, y_test

    def run_trials(self, X, y, N):
        """
        Run N trials & aggregate results the same as in the cross validation.
        :param X: DataFrame of features data
        :param y: DataFram with TARGET_LABEL column
        :param N: Number of trials
        """

        results = []
        for i in range(N):
            print("\n\nTrial:  " + str(i))
            X_train, y_train, X_test, y_test = self.manually_stratify(X, y, .66)

            # Oversample training data only
            if self.oversample is not None:
                X_train, y_train = self.super_sample(X_train, y_train)

            # Scale and apply PCA
            X_train, X_test = self.scale_data(X_train, X_test)
            if self.pca is not None:
                X_train, X_test = self.apply_PCA(X_train, X_test)

            # Train
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

    def compute_probability_range_metrics(self, results, bin_size=0.1):
        """
        Computes True Positive & Total metrics, split by >= probability assigned to class for cumulative ranges dictated by bin_size. Used to plot probability assigned vs completeness (TP/total, per bin).
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :param bin_size: Size of each bin (range of probabilities) to consider at a time; must be betwen 0 and 1
        :return range_metrics: Map of classes to [TP_range_sums, total_range_sums]
            total_range_sums: # of samples with probability in range for this class
            TP_range_sums: true positives per range
        """

        results = np.concatenate(results)

        range_metrics = {}
        label_index = len(self.class_labels)  # Last column is label
        for class_index, class_name in enumerate(self.class_labels):
            tp_probabilities = []  # probabilities for True Positive samples
            total_probabilities = []
            for row in results:
                labels = row[label_index]

                # Sample is an instance of this current class.
                is_class = self.is_class(class_name, labels)

                # Get class index of max prob; exclude last column since it is label
                max_class_prob = np.max(row[: len(row) - 1])
                max_class_index = np.argmax(row[: len(row) - 1])
                max_class_name = self.class_labels[max_class_index]

                if is_class and max_class_name == class_name:
                    tp_probabilities.append(max_class_prob)

                total_probabilities.append(row[class_index])

            # left inclusive, first bin is 0 <= x < .1. ; except last bin <=1
            bins = np.arange(0, 1 + bin_size, bin_size)
            tp_range_counts = np.histogram(tp_probabilities, bins=bins)[0].tolist()
            total_range_counts = np.histogram(total_probabilities, bins=bins)[0].tolist()

            range_metrics[class_name] = [tp_range_counts, total_range_counts]
        return range_metrics

    def compute_metrics(self, results):
        """
        Compute TP, FP, TN, and FN per class. Each sample is assigned its lowest-level class hierarchy label as its label. This is important, otherwise penalties will go across classes.
        :param results: List of 2D Numpy arrays with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :return class_metrics: Map from class name to map of {"TP": w, "FP": x, "FN": y, "TN": z}
        """
        # results = np.concatenate(results)

        # Last column is label
        label_index = len(self.class_labels)
        class_metrics = {cn: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                         for cn in self.class_labels}

        set_totals = {cn: {fold: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                           for fold in range(len(results))} for cn in self.class_labels}

        for class_name in self.class_labels:
            for index, result_set in enumerate(results):
                for row in result_set:
                    labels = row[label_index]
                    # Sample is an instance of this current class.
                    is_class = self.is_class(class_name, labels)
                    # Get class index of max prob; exclude last column since it is label
                    max_class_index = np.argmax(row[:len(row) - 1])
                    max_class_name = self.class_labels[max_class_index]

                    if class_name == max_class_name:
                        if is_class:
                            class_metrics[class_name]["TP"] += 1
                            set_totals[class_name][index]["TP"] += 1
                        else:
                            class_metrics[class_name]["FP"] += 1
                            set_totals[class_name][index]["FP"] += 1
                    else:
                        if is_class:
                            class_metrics[class_name]["FN"] += 1
                            set_totals[class_name][index]["FN"] += 1
                        else:
                            class_metrics[class_name]["TN"] += 1
                            set_totals[class_name][index]["TN"] += 1

        return class_metrics, set_totals

    def compute_baselines(self, class_counts, y):
        """
        Get random classifier baselines for recall, specificity (negative recall), and precision
        """
        pos_baselines = {}
        neg_baselines = {}
        precision_baselines = {}
        total_count = y.shape[0]
        class_priors = {c: 1 / len(self.class_labels)
                        for c in self.class_labels}

        for class_name in self.class_labels:
            # Compute baselines
            class_freq = class_counts[class_name] / total_count
            pos_baselines[class_name] = class_priors[class_name]
            neg_baselines[class_name] = (1 - class_priors[class_name])
            precision_baselines[class_name] = class_freq

        return pos_baselines, neg_baselines, precision_baselines

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

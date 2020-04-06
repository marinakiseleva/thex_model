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

# local imports
from thex_data.data_init import *
from thex_data.data_filter import filter_data
from thex_data.data_prep import get_source_target_data
from thex_data.data_transform import scale_data, apply_PCA
from thex_data.data_plot import *
from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS, class_to_subclass as CLASS_HIERARCHY
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
        sys.stdout = open(self.dir + "/experiment.log", "a")

        # Must add Unspecifieds to tree, so that when searching for lowest-level
        # label, the UNDEF one returns.
        self.class_hier = CLASS_HIERARCHY.copy()
        for parent in CLASS_HIERARCHY.keys():
            self.class_hier[parent].insert(0, UNDEF_CLASS + parent)
        self.tree = util.init_tree(self.class_hier)
        self.class_levels = util.assign_levels(self.tree, {}, self.tree.root, 1)
        data_filters = {'cols': None,  # Names of columns to filter on; default is all numeric cols
                        'col_matches': None,  # String by which columns will be selected
                        'num_runs': None,  # Number of trials
                        'folds': None,  # Number of folds if using k-fold Cross Validation
                        'transform_features': True,  # Derive mag colors
                        'min_class_size': 9,
                        'max_class_size': None,
                        'pca': None,  # Number of principal components
                        'class_labels': None,
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
        y = util.add_unspecified_labels_to_data(y, self.class_levels)

        self.class_labels = self.get_class_labels(
            data_filters['class_labels'], y, data_filters['min_class_size'])

        # Pre-processing dependent on class labels
        X, y = filter_data(X, y, data_filters, self.class_labels, self.class_hier)

        # Save relevant data attributes to self
        self.X = X
        self.y = y
        self.num_folds = data_filters['folds']
        self.num_runs = data_filters['num_runs']
        self.pca = data_filters['pca']
        self.nb = data_filters['nb']
        self.class_counts = self.get_class_counts(y)

        print("\nClasses Used:\n" + str(self.class_labels))
        print("\nFeatures Used:\n" + str(list(X)))
        print("\nClass counts:\n" + str(self.class_counts))

    def run_model(self):
        """
        Visualize data, run analysis, and record results in self.results, a list of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """
        print("\nRunning " + str(self.name))
        self.visualize_data(self.X, self.y)

        if self.num_runs is not None:
            self.results = self.run_trials(self.X, self.y, self.num_runs)
        elif self.num_folds is not None:
            self.results = self.run_cfv(self.X, self.y)
        else:
            raise ValueError("Must pass in either number of folds or runs.")

        # Save results in pickle
        with open(self.dir + '/results.pickle', 'wb') as f:
            pickle.dump(self.results, f)
        with open(self.dir + '/y.pickle', 'wb') as f:
            pickle.dump(self.y, f)
        self.visualize_performance()

        sys.stdout = sys.__stdout__

    def relabel_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = util.convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def get_class_labels(self, user_defined_labels, y, N):
        """
        Keep all classes with # of samples > min class size. If a class has only 1 child class that meets this criteria, we only use the parent class and not the children. We ensure each class is greater than N, and then we make disjoint by keeping only leaves of the new class hierarchy tree. If user_defined_labels is not NULL, we filter on this at the end.
        :param user_defined_labels: None, or valid list of class labels
        :param y: DataFrame with TARGET_LABEL column
        :param N: Minimum # of samples per class to keep it
        """
        # Return classes passed in by user if there are any
        if user_defined_labels is not None:
            return user_defined_labels

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
        init_plot_settings()

        completeness = calculate_completeness(X, y, self.class_labels)
        ordered_comp = self.get_ordered_metrics(completeness)
        visualize_completeness(self.dir, X, ordered_comp[0], ordered_comp[1])

        ordered_counts = self.get_ordered_metrics(self.class_counts)

        plot_class_hist(self.dir, ordered_counts[0], ordered_counts[1])
        # Combine X and y for plotting feature dist
        df = pd.concat([X, y], axis=1)
        features = list(df)
        if 'redshift' in features:
            plot_feature_distribution(self.dir,  df, 'redshift',
                                      self.class_labels, self.class_counts)

    def visualize_performance(self):
        """
        Visualize performance

        """
        range_metrics = self.compute_probability_range_metrics(self.results)
        self.plot_prob_pr_curves(range_metrics, self.class_counts)
        self.plot_probability_vs_class_rates(range_metrics)
        class_metrics, set_totals = self.compute_metrics(self.results)
        self.plot_all_metrics(class_metrics, set_totals, self.y)

        return -1

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

            # Scale and apply PCA
            X_train, X_test = scale_data(X_train, X_test)
            if self.pca is not None:
                X_train, X_test = apply_PCA(X_train, X_test, self.pca)

            # Train
            self.train_model(X_train, y_train)

            # Test model
            probabilities = self.get_all_class_probabilities(X_test)
            # Add labels as column to probabilities, for later evaluation
            label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
            probabilities = np.hstack((probabilities, label_column))
            results.append(probabilities)

        return results

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
            print("\n\nTrial:  " + str(i + 1))
            X_train, y_train, X_test, y_test = self.manually_stratify(X, y, .66)

            # Scale and apply PCA
            X_train, X_test = scale_data(X_train, X_test)
            if self.pca is not None:
                X_train, X_test = apply_PCA(X_train, X_test, self.pca)

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
        Returns map of class name to true positives & total count per probability bin. Also saves probability rates and # of samples in each class to maps (self.class_prob_rates, self.class_positives)
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :param bin_size: Size of each bin (range of probabilities) to consider at a time; must be betwen 0 and 1
        :return range_metrics: Map of classes to [TP_range_sums, total_range_sums]
            total_range_sums: # of samples with probability in range for this class
            TP_range_sums: true positives per range
        """

        results = np.concatenate(results)

        range_metrics = {}
        label_index = len(self.class_labels)  # Last column is label

        self.class_prob_rates = {}
        self.class_positives = {}
        for class_index, class_name in enumerate(self.class_labels):
            tp_probabilities = []  # probabilities for True Positive samples
            pos_probabilities = []  # probabilities for Positive samples
            total_probabilities = []
            for row in results:
                labels = row[label_index]

                # Sample is an instance of this current class.
                is_class = self.is_class(class_name, labels)

                # Get class index of max prob; exclude last column since it is label
                max_class_prob = np.max(row[: len(row) - 1])
                max_class_index = np.argmax(row[: len(row) - 1])
                max_class_name = self.class_labels[max_class_index]

                if is_class:
                    pos_probabilities.append(row[class_index])

                if is_class and max_class_name == class_name:
                    tp_probabilities.append(max_class_prob)

                total_probabilities.append(row[class_index])

            # left inclusive, first bin is 0 <= x < .1. ; except last bin <=1
            bins = np.arange(0, 1 + bin_size, bin_size)
            tp_range_counts = np.histogram(tp_probabilities, bins=bins)[0].tolist()
            total_range_counts = np.histogram(total_probabilities, bins=bins)[0].tolist()

            range_metrics[class_name] = [tp_range_counts, total_range_counts]

            # Calculate class prob rates separately
            pos_class_per_range = np.histogram(pos_probabilities, bins=bins)[0].tolist()
            self.class_positives[class_name] = pos_class_per_range
            class_prob_rates = np.array(pos_class_per_range) / \
                np.array(total_range_counts)
            class_prob_rates[np.isinf(class_prob_rates)] = 0
            class_prob_rates[np.isnan(class_prob_rates)] = 0
            self.class_prob_rates[class_name] = class_prob_rates

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

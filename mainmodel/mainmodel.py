"""
Base model structure which models all build off of. Model structure allows for flexibility in normalization method and hierarchy use. Model defines structure that all submodels have in common:

Initial data collection class
Evaluation strategy (k-fold cross validation or a set number of trials)
Performance aggregation

Subclasses differ in how they normalize across the class hierarchy.

"""
import sys
import random
import pickle
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# local imports
from mainmodel.helper_compute import get_ordered_metrics, compute_performance
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
        init_plot_settings()
        print("Saving " + self.name + " output to directory " + self.dir)

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
                        'nb': False,  # Naive Bayes multiclass
                        'priors': None  # Priors in order of class_labels
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

        self.class_labels = self.get_class_labels(y,
                                                  data_filters['min_class_size'])
        # Pre-processing dependent on class labels
        if data_filters['class_labels'] is not None:
            self.class_labels = data_filters['class_labels']

        X, y = filter_data(X, y, data_filters, self.class_labels, self.class_hier)

        # Save relevant data attributes to self
        self.X = X
        self.y = y
        self.class_priors = data_filters['priors']
        self.num_folds = data_filters['folds']
        self.num_runs = data_filters['num_runs']
        self.pca = data_filters['pca']
        self.nb = data_filters['nb']
        self.transform_features = data_filters['transform_features']
        self.class_counts = self.get_class_counts(y)
        self.normalize = True

        print("\nClasses Used:\n" + str(self.class_labels))
        print("\nClass Priors:\n" + str(self.class_priors))
        print("\nFeatures Used:\n" + str(list(X)))
        print("\nClass Counts:\n" + str(self.class_counts))

        cc = sum(self.class_counts.values())
        a = self.y.shape[0]
        if cc != a:
            raise ValueError("Data is not filtered properly.")

    def run(self):
        """
        Run model, either by trials or cross fold validation
        """
        if self.num_runs is not None:
            self.results = self.run_trials(self.X, self.y, self.num_runs)
        elif self.num_folds is not None:
            self.results = self.run_cfv(self.X, self.y)
        else:
            raise ValueError("Must pass in either number of folds or runs.")

    def analyze_probabilities(self):
        """
        Evalaute how well the KDEs fit the data by visualizing the performance at different proportions of top unnormalized probabilities (densities). 
        """
        self.normalize = False
        self.run()
        results = np.concatenate(self.results)
        with open(self.dir + '/density_results.pickle', 'wb') as f:
            pickle.dump(results, f)

        self.plot_density_performance(results)
        self.plot_density_half_compare(results)

    def run_model(self):
        """
        Visualize data, run analysis, and record results in self.results, a list of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        """
        print("\nRunning " + str(self.name))

        self.visualize_data()

        self.run()

        # Save results in pickle
        with open(self.dir + '/results.pickle', 'wb') as f:
            pickle.dump(self.results, f)
        with open(self.dir + '/y.pickle', 'wb') as f:
            pickle.dump(self.y, f)

        self.visualize_performance()

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

    def get_class_labels(self, y, N):
        """
        Keep all classes with # of samples > N. If a class has only 1 child class that meets this criteria, we only use the parent class and not the children. We ensure each class is greater than N, and then we make disjoint by keeping only leaves of the new class hierarchy tree. 
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

    def visualize_data(self):
        """
        Visualize data completeness and distribution
        """
        X = self.X.copy()
        y = self.y.copy()

        completeness = calculate_completeness(X, y, self.class_labels)
        ordered_comp = get_ordered_metrics(completeness)

        visualize_completeness(self.dir, X, ordered_comp[0], ordered_comp[1])

        ordered_counts = get_ordered_metrics(self.class_counts)
        ordered_names = ordered_counts[0]
        ordered_metrics = ordered_counts[1]
        plot_class_hist(self.dir, ordered_names, ordered_metrics)
        # Combine X and y for plotting feature dist
        df = pd.concat([X, y], axis=1)
        features = list(df)
        if 'redshift' in features:
            plot_feature_distribution(model_dir=self.dir,
                                      df=df.copy(),
                                      feature='redshift',
                                      class_labels=self.class_labels)

    def visualize_performance(self):
        """
        Visualize performance
        """
        self.plot_confusion_matrix(self.results)
        range_metrics = self.compute_probability_range_metrics(
            self.results)
        self.plot_prob_pc_curves(range_metrics, self.class_counts)
        self.plot_probability_vs_class_rates(range_metrics)

        N = self.num_runs if self.num_runs is not None else self.num_folds

        pc_per_trial = self.get_pc_per_trial(self.results)
        ps, cs = self.get_avg_pc(pc_per_trial, N)

        self.plot_all_metrics(ps, cs, pc_per_trial, self.y)

    def run_cfv(self, X, y):
        """
        Run k-fold cross validation
        Number of folds comes from self.num_folds
        :param X: DataFrame of features data
        :param y: DataFram with TARGET_LABEL column
        """
        kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True)
        results = []
        self.datas = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index].reset_index(
                drop=True), X.iloc[test_index].reset_index(drop=True)
            y_train, y_test = y.iloc[train_index].reset_index(
                drop=True), y.iloc[test_index].reset_index(drop=True)

            # Scale and apply PCA
            if self.transform_features:
                X_train, X_test = scale_data(X_train, X_test)
            if self.pca is not None:
                X_train, X_test = apply_PCA(X_train, X_test, self.pca)

            # Train
            self.train_model(X_train, y_train)
            self.datas.append(X_test)

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
            X_train, y_train, X_test, y_test = self.manually_stratify(X, y, .8)

            # Scale and apply PCA
            if self.transform_features:
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

    def get_num_classes(self):
        """
        Return number of classes
        """
        return len(self.class_labels)

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

    def is_class(self, class_name, labels):
        """
        Boolean which returns True if class name is in the list of labels, and False otherwise.
        """
        if class_name in util.convert_str_to_list(labels):
            return True
        else:
            return False

    def is_true_positive(self, is_class, row, class_index,  max_class_name, class_name):
        """
        Determines if the prediction is a true positive for the class class_name (need this to overwrite in Binary)
        """
        return is_class and max_class_name == class_name

    def get_true_pos_prob(self, row, class_index, max_class_prob):
        """
        Get true positive probability (need this to overwrite in Binary)
        """
        return max_class_prob

    def compute_probability_range_metrics(self, results, bin_size=0.1, concat=True):
        """
        Returns map of class name to true positives & total count per probability bin. Also saves probability rates and # of samples in each class to maps (self.class_prob_rates, self.class_positives)
        :param results: List of 2D Numpy arrays, with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :param bin_size: Size of each bin (range of probabilities) to consider at a time; must be betwen 0 and 1
        :return range_metrics: Map of classes to [TP_range_sums, total_range_sums]
            total_range_sums: # of samples with probability in range for this class
            TP_range_sums: true positives per range
        """
        if concat:
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

                is_tp = self.is_true_positive(
                    is_class, row, class_index,  max_class_name, class_name)
                if is_tp:
                    pos_prob = self.get_true_pos_prob(row, class_index, max_class_prob)
                    tp_probabilities.append(pos_prob)

                total_probabilities.append(row[class_index])

            # left inclusive, first bin is 0 <= x < .1. ; except last bin <=1
            bins = np.arange(0, 1.01, bin_size)
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
        Compute TP, FP, TN, and FN per class and (if as_sets is True) compute those metrics for each class for each fold/run.
        :param results: List of 2D Numpy arrays with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels & the last column containing the full, true label
        :return class_metrics: Map from class name to map of {"TP": w, "FP": x, "FN": y, "TN": z}
        """
        class_metrics = {cn: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                         for cn in self.class_labels}
        for row in results:
            class_metrics = self.get_row_metrics(row, class_metrics)
        return class_metrics

    def get_pc_per_trial(self, results):
        """
        Get purity and completeness per trial/fold, per class
        Return [[c1,p1], [c2,p2] ..., [cN, pN]] where each c and p is a map from class names to completeness or purity, respectively
        """
        t_performances = []  # performance per trial/fold
        for index, trial in enumerate(results):
            class_metrics = {cn: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                             for cn in self.class_labels}
            # For each predicted row
            for row in trial:
                class_metrics = self.get_row_metrics(row, class_metrics)
            # Compute purity & completeness for this trial/fold (per class)
            puritys, comps = compute_performance(class_metrics)
            t_performances.append([puritys, comps])

            print("Metrics for trial " + str(index + 1))
            print("Purity: " + str(puritys))
            print("Completeness: " + str(comps))

        return t_performances

    def get_avg_pc(self, t_performances, N):
        """
        Compute average purity & completeness over folds/trials per class
        :param t_performances: Return of self.get_pc_per_trial
        :param N: Number of trials/folds
        """
        # Average purity & completeness over folds/trials (per class)
        avg_comps = {cn: 0 for cn in self.class_labels}
        avg_purities = {cn: 0 for cn in self.class_labels}
        for class_name in self.class_labels:
            p_N = N  # N values for purity
            # Aggregate over trials/folds
            for p, c in t_performances:
                if p[class_name] is None:
                    # No purity for this trial -> exclude from average
                    print("No measurable purity for " + class_name)
                    p_N = p_N - 1
                else:
                    avg_purities[class_name] += p[class_name]

                avg_comps[class_name] += c[class_name]
            # Get average
            avg_purities[class_name] = avg_purities[class_name] / p_N
            avg_comps[class_name] = avg_comps[class_name] / N

        return avg_purities, avg_comps

    def get_row_metrics(self, row, class_metrics, index=None):
        """
        Helper function for compute_metrics
        Get TP, FP, TN, FN metrics for this row & update in passed in dict of class_metrics. 
        :param index: Index of result set
        :param row: Numpy row of probabiliites (last col is label)
        :param class_metrics: Dict to update
        """
        # Last column is label
        label_index = len(self.class_labels)
        labels = row[label_index]

        for class_name in self.class_labels:
            # Sample is an instance of this current class.
            is_class = self.is_class(class_name, labels)
            # Get class index of max prob; exclude last column since it is label
            max_class_index = np.argmax(row[:len(row) - 1])
            max_class_name = self.class_labels[max_class_index]

            if class_name == max_class_name:
                if is_class:
                    class_metrics[class_name]["TP"] += 1
                else:
                    class_metrics[class_name]["FP"] += 1
            else:
                if is_class:
                    class_metrics[class_name]["FN"] += 1
                else:
                    class_metrics[class_name]["TN"] += 1
        return class_metrics

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

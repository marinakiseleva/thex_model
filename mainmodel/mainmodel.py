"""
Base model structure which models all build off of. Model structure allows for flexibility in normalization method and hierarchy use. Model defines structure that all submodels have in common:

Initial data collection class
Evaluation strategy (k-fold cross validation or a set number of trials)
Performance aggregation

Subclasses differ in how they normalize across the class hierarchy.

"""
import sys
import random
import math
import pickle
from abc import ABC, abstractmethod
import pandas as pd
import copy
import time


# local imports
from mainmodel.helper_compute import get_ordered_metrics
from thex_data.data_init import *
from thex_data.data_filter import filter_data
from thex_data.data_prep import get_source_target_data
from thex_data.data_transform import scale_data, apply_PCA
from thex_data.data_plot import *
from thex_data.data_consts import DATA_PATH, TARGET_LABEL, UNDEF_CLASS, CLASS_HIERARCHY
from mainmodel.vis import MainModelVisualization
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
        self.class_hier = copy.deepcopy(CLASS_HIERARCHY)
        for parent in CLASS_HIERARCHY.keys():
            self.class_hier[parent].insert(0,  UNDEF_CLASS + parent)
        self.tree = util.init_tree(self.class_hier)
        self.class_levels = util.assign_levels(self.tree, {}, self.tree.root, 1)
        # Default filter values
        data_filters = {'cols': None,  # Names of columns to filter on; default is all numeric cols
                        'col_matches': None,  # String by which columns will be selected
                        'num_runs': None,  # Number of trials
                        'folds': None,  # Number of folds if using k-fold Cross Validation
                        'transform_features': True,  # Derive mag colors & scale
                        'min_class_size': 40,
                        'max_class_size': None,
                        'pca': None,  # Number of principal components
                        'class_labels': None,
                        'data': None,  # [X,y] data, as PD dataframes.
                        'nb': False,  # Naive Bayes multiclass
                        'priors': False,  # Priors, boolean
                        'data_file': DATA_PATH,  # Default data file used
                        'linear_calib': False,
                        'lsst_test': False,  # if True, groups Ib/c, Ib, and Ic as Ib/c
                        'Zmodel': False,
                        'balanced_purity': False,
                        'case_code': None
                        }
        # Use default for filters not passed in.
        for data_filter in user_data_filters.keys():
            data_filters[data_filter] = user_data_filters[data_filter]

        # Use passed-in data, if available
        if data_filters['data'] is not None:
            X = data_filters['data'][0]
            y = data_filters['data'][1]
        else:
            # list of features to use from database
            print("Using data: " + data_filters['data_file'])
            features = collect_cols(data_filters['cols'], data_filters[
                                    'col_matches'], data_filters['data_file'])
            X, y = get_source_target_data(features, data_filters)

            # Redefine labels with Unspecifieds
            y = util.add_unspecified_labels_to_data(y, self.class_levels)

        self.class_labels = self.get_class_labels(y,
                                                  data_filters['min_class_size'])
        # Pre-processing dependent on class labels
        if data_filters['class_labels'] is not None:
            self.class_labels = copy.deepcopy(data_filters['class_labels'])

        if data_filters['lsst_test']:
            self.class_labels = ["Unspecified Ia", "II",
                                 "Ia-91bg", "TDE", "Ic", "Ib/c", "Ib", "SLSN-I"]

        X, y = filter_data(X, y, data_filters, self.class_labels, self.class_hier)

        # Aggregate Ibc class if making LSST-like.
        if data_filters['lsst_test']:
            y, self.class_labels = util.group_labels(
                y, self.class_labels, ["Ib", "Ic", "Ib/c"], "Ibc")

        # Use only redshift as a feature in Zmodel.
        if data_filters['Zmodel']:
            X = X[['redshift']]

        self.X = X
        self.y = y
        self.num_folds = data_filters['folds']
        self.num_runs = data_filters['num_runs']
        self.pca = data_filters['pca']
        self.nb = data_filters['nb']
        self.transform_features = data_filters['transform_features']
        self.linear_calib = data_filters['linear_calib']
        self.class_counts = self.get_class_counts(y)
        self.normalize = True
        self.training_lls = None
        self.balanced_purity = data_filters['balanced_purity']

        # Print helpful info about this model's run.
        print("\nClasses:\n" + str(self.class_labels))
        if data_filters['priors'] == True:
            self.class_priors = {}
            total = sum(self.class_counts.values())
            for c in self.class_labels:
                self.class_priors[c] = round(self.class_counts[c] / total, 2)
                if self.class_priors[c] == 0:
                    # Lowest threshold is 1% prior
                    self.class_priors[c] = 0.001
            print("\nClass Priors:\n" + str(self.class_priors))
        else:
            self.class_priors = None

        print("\nFeatures:\n" + str(list(X)))
        util.pretty_print_dict(self.class_counts, "Class Counts")

        cc = sum(self.class_counts.values())
        a = self.y.shape[0]
        if cc != a:
            raise ValueError("Data is not filtered properly.")

    def run(self, start_time=None):
        """
        Run model, either by trials or cross fold validation
        """
        if self.linear_calib:
            if self.num_folds != 3:
                raise ValueError("Can only do linear calibration w/ 3 folds.")
            self.results = self.run_3cfv_calib(start_time)
        elif self.num_runs is not None:
            self.results = self.run_trials(self.X, self.y, self.num_runs)
        elif self.num_folds is not None:
            self.results = self.run_cfv(start_time)
        else:
            raise ValueError("Must pass in either number of folds or runs.")

    def run_density_analysis(self):
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
        with open(self.dir + '/y.pickle', 'wb') as f:
            pickle.dump(self.y, f)
        self.visualize_data()

        start = time.time()
        self.run(start)
        end = time.time()
        print(util.get_runtime(start, end))

        # Save results in pickle
        with open(self.dir + '/results.pickle', 'wb') as f:
            pickle.dump(self.results, f)

        if self.training_lls is not None:
            avg_lls = {cn: 0 for cn in self.class_labels}
            for i_map in self.training_lls:
                for class_name in self.class_labels:
                    avg_lls[class_name] += i_map[class_name]

            for cn in self.class_labels:
                avg_lls[cn] = round(avg_lls[cn] / len(self.training_lls), 2)

            util.pretty_print_dict(
                avg_lls, "Average log-likelihood fit over training data: ")

        self.visualize_performance()

    def relabel_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        Used in Binary and OVA models.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            label = 1 if self.is_class(class_name, row[TARGET_LABEL]) else 0
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
        new_hier = copy.deepcopy(self.class_hier)

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
                if self.is_class(class_name, row[TARGET_LABEL]):
                    class_counts[class_name] += 1
        return class_counts

    def visualize_data(self):
        """
        Visualize data completeness and distribution
        """
        X = self.X.copy(deep=True)
        y = self.y.copy(deep=True)

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
            plot_feature_hist(model_dir=self.dir,
                              df=df.copy(deep=True),
                              feature='redshift',
                              class_labels=self.class_labels)

    def visualize_performance(self):
        """
        Visualize performance
        """
        N = self.num_runs if self.num_runs is not None else self.num_folds
        pc_per_trial = self.get_pc_per_trial(self.results)
        ps, cs = self.get_pc_performance(pc_per_trial)

        self.plot_all_metrics(ps, cs, pc_per_trial, self.y)

        self.plot_confusion_matrix(self.results)
        range_metrics = self.compute_probability_range_metrics(
            self.results)
        self.plot_prob_pc_curves(range_metrics)
        range_metrics = self.compute_probability_range_metrics(
            self.results, bin_size=0.2)
        self.plot_probability_vs_class_rates(range_metrics)

    def kfold_stratify(self):
        """
        Stratify into folds, with equal class representation in each fold
        """
        fold_indices = {x: [] for x in range(self.num_folds)}
        for class_name in self.class_labels:
            class_indices = []
            for index, row in self.y.iterrows():
                if self.is_class(class_name, row[TARGET_LABEL]):
                    class_indices.append(index)

            fold_count = math.floor(len(class_indices) / self.num_folds)
            excess = len(class_indices) - (self.num_folds * fold_count)
            for i in range(self.num_folds):
                a = random.sample(class_indices, fold_count)
                if excess > 0:
                    sampled = random.sample(class_indices, fold_count + 1)
                    excess -= 1
                else:
                    sampled = random.sample(class_indices, fold_count)
                class_indices = [x for x in class_indices if x not in sampled]

                fold_indices[i] += sampled

        return fold_indices

    def calibrate_probabilities_linear(self, test_probs, train_probs):
        """
        Calibrate probabilities by fitting a line to the training probs.  
        :param train_probs: 2D numpy array with probabilities per row in order
        of self.class_labels, and last col is label
        :param test_probs: same structure as train_probs; but it is these probs which we are going to calibrate, based on the empirical probs from train_probs
        :param train_results: probabilities assigned to training data
        """

        # Pass in training data & labels as 2D numpy array
        train_metrics = self.compute_probability_range_metrics(
            train_probs,  bin_size=0.1, concat=False)
        # train_metrics is dict from class name to
        # [tp_range_counts, total_range_counts]
        print("Validation Metrics")
        for cn in train_metrics.keys():
            print(cn)
            print(train_metrics[cn])

        # Recalibrate test_probs based on those metrics
        row_index = 0
        # Get linear calibrator per class
        calibs = {}
        for class_index, class_name in enumerate(self.class_labels):
            TPs, totals = train_metrics[class_name]
            emp_probs = []
            for index, tp in enumerate(TPs):
                if totals[index] == 0:
                    ratio = 0
                else:
                    ratio = tp / totals[index]
                emp_probs.append(ratio)
            x = np.arange(0.05, 1, 0.1)
            z = np.polyfit(x, np.array(emp_probs), 1)
            p = np.poly1d(z)
            calibs[class_name] = p

        row_index = 0
        for row in test_probs:
            # print("old row " + str(row))
            for class_index, class_name in enumerate(self.class_labels):
                class_f = calibs[class_name]
                new_prob = class_f(row[class_index])
                if new_prob < 0:
                    new_prob = 0
                elif new_prob > 1:
                    new_prob = 1
                row[class_index] = new_prob
            # Normalize probs in row so it sums to 1
            row_sum = sum(row[:-1])
            if row_sum == 0:
                test_probs[row_index, :-1] = 0
            else:
                test_probs[row_index, :-1] = row[:-1] / row_sum
            row_index += 1

        return test_probs

    def calibrate_probabilities_bin(self, test_probs, train_probs):
        """
        Calibrate probabilities using binning; assigning probability based on empirical probability from training data.  
        :param test_probs: 2D numpy array with probabilities per row in order
        of self.class_labels, and last col is label
        :param train_results: probabilities assigned to training data
        """

        print("\nCalibrating probabilities ")

        # Pass in training data & labels as 2D numpy array
        train_metrics = self.compute_probability_range_metrics(
            train_probs,  bin_size=0.1, concat=False)

        print("Training Metrics")
        for cn in train_metrics.keys():
            print(cn)
            print(train_metrics[cn])

        orig_test_metrics = self.compute_probability_range_metrics(
            test_probs,  bin_size=0.1, concat=False)

        # train_metrics is dict from class name to
        # [tp_range_counts, total_range_counts]

        # Recalibrate test_probs based on those metrics
        row_index = 0
        for row in test_probs:
            for class_index, class_name in enumerate(self.class_labels):
                TPs, totals = train_metrics[class_name]
                class_prob = row[class_index]
                # bins are left inclusive,
                # so first bin is 0 <= x < .1. ; so last bin <=1.01
                # Find what bin the assigned prob is in
                if class_prob == 1:
                    bin_i = 9
                else:
                    bin_i = math.floor(class_prob * 10)
                if totals[bin_i] == 0:
                    cal_prob = 0
                else:
                    # get empirical probability of this bin
                    cal_prob = TPs[bin_i] / (totals[bin_i] + 0.0)

                row[class_index] = cal_prob
            # Normalize probs in row so it sums to 1
            row_sum = sum(row[:-1])
            if row_sum == 0:
                test_probs[row_index, :-1] = 0
            else:
                test_probs[row_index, :-1] = row[:-1] / row_sum
            row_index += 1

        return test_probs

    def run_3cfv_calib(self, start_time):
        """
        Run 3-fold cross validation, using one fold for training, one for validation, and one for testing - and calibrate using validation fold. 
        Number of folds comes from self.num_folds
        :param X: DataFrame of features data
        :param y: DataFram with TARGET_LABEL column
        """
        fold_indices = self.kfold_stratify()
        results = []
        train_results = []
        self.datas = []
        train_folds = [0, 1, 2]
        test_folds = [1, 2, 0]
        val_folds = [2, 0, 1]
        for i in range(3):
            print("\nFold " + str(i + 1) + "\t" +
                  util.get_runtime(start_time, time.time()))
            # Select training, val, and test data
            train_fold = train_folds[i]
            X_train = self.X.iloc[fold_indices[train_fold]].reset_index(
                drop=True)
            y_train = self.y.iloc[fold_indices[train_fold]].reset_index(
                drop=True)

            test_fold = test_folds[i]
            X_test = self.X.iloc[fold_indices[test_fold]].reset_index(
                drop=True)
            y_test = self.y.iloc[fold_indices[test_fold]].reset_index(
                drop=True)

            val_fold = val_folds[i]
            X_val = self.X.iloc[fold_indices[val_fold]].reset_index(
                drop=True)
            y_val = self.y.iloc[fold_indices[val_fold]].reset_index(
                drop=True)

            # Scale and apply PCA
            if self.transform_features:
                X_train, X_test, X_val = scale_data(X_train, X_test, X_val)
            if self.pca is not None:
                X_train, X_test = apply_PCA(X_train, X_test, self.pca)

            # Train
            self.train_model(X_train, y_train)
            self.datas.append(X_test)

            # Get probabilities and labels for training, validation, and test data
            train_probs = self.get_all_class_probabilities(X_train)
            # Add labels as column to probabilities, for later evaluation
            train_labels = y_train[TARGET_LABEL].values.reshape(-1, 1)
            train_probs = np.hstack((train_probs, train_labels))

            # Val model
            val_probs = self.get_all_class_probabilities(X_val)
            label_column = y_val[TARGET_LABEL].values.reshape(-1, 1)
            val_probs = np.hstack((val_probs, label_column))

            # Test model
            test_probs = self.get_all_class_probabilities(X_test)
            label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
            test_probs = np.hstack((test_probs, label_column))

            # Calibrate probabilities
            print("\nCalibrating probabilities using validation data ")
            test_probs = self.calibrate_probabilities_linear(
                test_probs=test_probs, train_probs=val_probs)

            results.append(test_probs)

        return results

    def run_cfv(self, start_time):
        """
        Run k-fold cross validation
        Number of folds comes from self.num_folds
        :param X: DataFrame of features data
        :param y: DataFram with TARGET_LABEL column
        """
        fold_indices = self.kfold_stratify()
        results = []
        train_results = []
        self.datas = []
        for i in range(self.num_folds):
            print("\nFold " + str(i + 1) + "\t" +
                  util.get_runtime(start_time, time.time()))
            # Select training and test indices for this fold
            indices = fold_indices[i]
            test_indices_X = self.X.index.isin(fold_indices[i])
            test_indices_y = self.y.index.isin(fold_indices[i])
            X_train = self.X[~test_indices_X].reset_index(
                drop=True)
            y_train = self.y[~test_indices_y].reset_index(
                drop=True)
            X_test = self.X.iloc[fold_indices[i]].reset_index(
                drop=True)
            y_test = self.y.iloc[fold_indices[i]].reset_index(
                drop=True)

            # Scale and apply PCA
            if self.transform_features:
                X_train, X_test = scale_data(X_train, X_test)
            if self.pca is not None:
                X_train, X_test = apply_PCA(X_train, X_test, self.pca)

            # Train
            self.train_model(X_train, y_train)

            # Save this to recalibrate probabilities
            train_probs = self.get_all_class_probabilities(X_train)
            train_labels = y_train[TARGET_LABEL].values.reshape(-1, 1)
            train_probs = np.hstack((train_probs, train_labels))

            # Test model
            test_probs = self.get_all_class_probabilities(X_test)
            # Add labels as column to probabilities, for later evaluation
            label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
            test_probs = np.hstack((test_probs, label_column))

            self.datas.append([X_test, test_probs])

            # # Calibrate probabilities
            # test_probs = self.calibrate_probabilities_linear(test_probs, train_probs)

            results.append(test_probs)

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
                if self.is_class(class_name, row[TARGET_LABEL]):
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

    def is_true_positive(self, row, eval_class_index):
        """
        Determines if the prediction is a true positive for the class eval_class (overriden Binary)
        :param row: Row of probabilities in order of self.class_labels, and last column is label, as list.
        :param eval_class_index: index of class for which to evaluate on. So if this is the name of the class with the max prob, return True, otherwise, False.
        """
        max_class_index = np.argmax(row[:len(row) - 1])
        max_class_name = self.class_labels[max_class_index]

        label_index = len(self.class_labels)
        labels = row[label_index]
        is_class = self.is_class(max_class_name, labels)

        # This is the max prob class, and its the one we are evaluating.
        eval_class = self.class_labels[eval_class_index]
        return is_class and max_class_name == eval_class

    def get_true_pos_prob(self, row, class_index, max_class_prob):
        """
        Get true positive probability (need this to overwrite in Binary)
        """
        return max_class_prob

    def compute_probability_range_metrics(self, results, bin_size=0.1, concat=True):
        """
        Returns map of class name to TPs & total count per probability bin. 
        Saves % of positives in each bin to self.class_prob_rates
        Saves # of positives in each bin to self.class_positives
        :param results: List of 2D Numpy arrays, with each row corresponding to sample,
            and each column the probability of that class,
            in order of self.class_labels & the last column containing the full, true label
        :param bin_size: Size of each bin; must be betwen 0 and 1
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
            all_probabilities = []
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

                is_tp = self.is_true_positive(row, class_index)
                if is_tp:
                    # TP - this class has max prob (or >0.5 for binary)
                    # Save the prob assigned to this class in TP list.
                    tp_probabilities.append(row[class_index])

                all_probabilities.append(row[class_index])

            # left inclusive, first bin is 0 <= x < .1. ; except last bin <=1
            bins = np.arange(0, 1.01, bin_size)
            tp_range_counts = np.histogram(tp_probabilities, bins=bins)[0].tolist()
            total_range_counts = np.histogram(all_probabilities, bins=bins)[0].tolist()

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

    def get_num_classes(self):
        """
        To override in binary classifier
        """
        return len(self.class_labels)

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

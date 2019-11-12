from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

from thex_data.data_clean import *
from thex_data.data_plot import *
from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, UNDEF_CLASS, PRED_LABEL, TREE_ROOT
from thex_data.data_prep import get_source_target_data
from thex_data.data_consts import class_to_subclass as hierarchy


from models.base_model.base_model import BaseModel
from models.base_model_mc.mc_base_model_performance import MCBaseModelPerformance
from models.base_model_mc.mc_base_model_plots import MCBaseModelVisualization


class MCBaseModel(BaseModel, MCBaseModelPerformance, MCBaseModelVisualization):
    """
    Abstract Class representing base functionality of all multiclass models. Inherits from BaseModel, and uses Mixins from other classes. Subclasses of models must implement all BaseModel abstract methods PLUS get_all_class_probabilities
    """
    @abstractmethod
    def get_all_class_probabilities(self, normalized):
        """
        Get class probability for each sample, for each class.
        :param normalized: Normalization technique
        :return: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        pass

    def run_model(self):
        """
        Override run_model in BaseModel.

        Set custom attributes for Multiclass classifiers: test_level (level of class hierarchy over which to get probabilities, for MCKDEModel), and class_labels (specific classes to run model on)
        """
        for parent in hierarchy.keys():
            hierarchy[parent].insert(0, UNDEF_CLASS + parent)
        self.tree = init_tree(hierarchy)
        # Root is level 1 in self.class_levels
        self.class_levels = assign_levels(self.tree, {}, self.tree.root, 1)
        self.level_classes = self.invert_class_levels(self.class_levels)
        self.test_level = self.user_data_filters[
            'test_level'] if 'test_level' in self.user_data_filters.keys() else None

        self.class_labels = self.user_data_filters[
            'class_labels'] if 'class_labels' in self.user_data_filters.keys() else None

        super(MCBaseModel, self).run_model()

    def test_model(self, keep_top_half=False):
        """
        OLD CODE. NEED TO DELETE.

        Get class prediction for each sample. If test_level is not None, save just the max probability class. If test_level is None, we need to consider probability for each level in the hierarchy, so we create a DataFrame with a column for each class in self.class_labels and save each probability.
        :return:
            when self.test_level is not None: DataFrame with column PRED_LABEL, which contains the class name of the class with the highest probability assigned.
            when self.test_level is None: DataFrame with column for each class in self.class_labels, and probabilities filled in.
        """
        # For diagnosing purposes - keep only top 1/2 probs
        if keep_top_half:
            unnormalized_max_probabilities = []
            for index, row in self.X_test.iterrows():
                unnormalized_probabilities = self.get_class_probabilities(
                    row)
                max_unnormalized_prob = max(unnormalized_probabilities.values())
                unnormalized_max_probabilities.append(max_unnormalized_prob)
            probs = np.array(unnormalized_max_probabilities)
            keep_indices = np.argwhere(probs > np.average(probs)).transpose()[0].tolist()

            # Get max class for those indices, and filter down self.X_test and
            # self.y_test to have same rows
            self.X_test = self.X_test.loc[self.X_test.index.isin(
                keep_indices)].reset_index(drop=True)
            self.y_test = self.y_test.loc[self.y_test.index.isin(
                keep_indices)].reset_index(drop=True)

        predictions = []
        for index, row in self.X_test.iterrows():
            probabilities = self.get_class_probabilities(row)
            if self.test_level is not None:
                # Single column with max prob class_name
                max_prob_class = max(probabilities, key=probabilities.get)
                predictions.append(max_prob_class)
            else:
                # Create column for each class in self.class_labels, w/ probability
                predictions.append(list(probabilities.values()))

        columns = [PRED_LABEL] if self.test_level is not None else self.class_labels
        predicted_classes = pd.DataFrame(predictions, columns=columns)

        return predicted_classes

    ##############################################################
    # Helper Functions

    def invert_class_levels(self, class_levels):
        """
        Convert dict of class names to levels to dict of class level # to list of classes (disjoint sets)
        """
        level_classes = {}
        for class_name in class_levels.keys():
            cur_level = class_levels[class_name]
            if cur_level in level_classes:
                level_classes[cur_level].append(class_name)
            else:
                level_classes[cur_level] = [class_name]
        return level_classes

    def get_parent_prob(self, class_name, probabilities):
        """
        Recurse up through tree, getting parent prob until we find a valid one. For example, there may only be CC, II, II P in CC so we need to inherit the probability of CC.
        """
        if class_name == TREE_ROOT:
            return 1
        elif self.tree._get_parent(class_name) in probabilities:
            return probabilities[self.tree._get_parent(class_name)]
        else:
            # Get next valid parent prob
            return self.get_parent_prob(self.tree._get_parent(class_name),
                                        probabilities)

    def get_class_counts(self, y):
        """
        Get number of samples for each class in self.class_labels
        :param y: DataFrame with TARGET_LABEL
        """
        class_counts = {n: 0 for n in self.class_labels}
        for df_index, row in y.iterrows():
            list_classes = convert_str_to_list(row[TARGET_LABEL])
            for class_name in list_classes:
                if class_name in self.class_labels:
                    class_counts[class_name] += 1
        return class_counts

    def get_parent_probabilities(self, class_counts):
        """
        Get probability of inner nodes given children frequencies.
        p(parent | child) = child count / sum of all children counts of parent
        :param class_counts: Map from class name to count
        """
        parent_probs = {}  # Map from (parent, child) tuples to probability
        for class_name in self.class_labels:
            children = self.tree._get_children(class_name)
            if len(children) > 0:
                parent_count = class_counts[class_name]
                for child in children:
                    if child in class_counts:
                        child_count = class_counts[child]
                        parent_probs[(class_name, child)] = child_count / parent_count

        return parent_probs

    ##############################################################
    # Data - Prepping Functionality

    def add_unspecified_labels_to_data(self, y):
        """
        Add unspecified label for each tree parent in data's list of labels
        """
        for index, row in y.iterrows():
            # Iterate through all class labels for this label
            max_depth = 0  # Set max depth to determine what level is undefined
            for label in convert_str_to_list(row[TARGET_LABEL]):
                if label in self.class_levels:
                    max_depth = max(self.class_levels[label], max_depth)
            # Max depth will be 0 for classes unhandled in hierarchy.
            if max_depth > 0:
                # Add Undefined label for any nodes at max depth
                for label in convert_str_to_list(row[TARGET_LABEL]):
                    if label in self.class_levels and self.class_levels[label] == max_depth:
                        add = ", " + UNDEF_CLASS + label
                        y.iloc[index] = y.iloc[index] + add
        return y

    def remove_excess_data(self, X, y):
        """
        Remove rows that do not contain data that is in self.class_labels. Keep if deepest-level node is in self.class_labels, or unspecified node in self.class_labels
        :param X: all features in DataFrame
        :param y: DataFrame of TARGET_LABEL column for all data, with unspecified labels added to TARGET_LABEL
        """
        # Keep rows that have an Unspecified label in self.class_labels.

        keep_indices = []
        for df_index, row in y.iterrows():
            list_classes = convert_str_to_list(row[TARGET_LABEL])
            keep = False
            deepest_level = 0
            deepest_class = None
            for c in list_classes:
                #  Maintain deepest class level
                if c in self.class_levels and self.class_levels[c] > deepest_level:
                    deepest_level = self.class_levels[c]
                    deepest_class = c
                if c in self.class_labels:
                    keep = True

            # Keep if deepest is in class_labels
            if deepest_class in self.class_labels:
                keep = True
            if keep:
                keep_indices.append(df_index)

        return X.loc[keep_indices, :].reset_index(drop=True), y.loc[keep_indices, :].reset_index(drop=True)

    def set_class_labels(self, y, user_defined_labels=None):

        if self.test_level is not None:
            # Gets all class labels on self.test_level,
            # including 'Undefined' classes of parent level.
            self.class_labels = []
            for class_name in self.get_mc_unique_classes(y):
                if class_name in self.class_levels:
                    class_level = self.class_levels[class_name]
                    if class_level == self.test_level:
                        self.class_labels.append(class_name)
                    elif class_level == self.test_level - 1:
                        self.class_labels.append(UNDEF_CLASS + class_name)
        else:
            self.class_labels = self.get_mc_unique_classes(y)

        if user_defined_labels is not None:
            # keep all undefined versions of user-defined classes
            for label in user_defined_labels:
                if UNDEF_CLASS + label in self.class_labels:
                    user_defined_labels.append(UNDEF_CLASS + label)
            self.class_labels = user_defined_labels

    def visualize_data(self, data_filters, y, X, class_counts):
        """
        Visualize distribution of data used to train and test
        """
        plot_class_hist(y, self.prep_file_name(self.name), True, class_counts)
        # Combine X and y for plotting feature dist
        df = pd.concat([X, y], axis=1)
        features = list(df)
        if 'redshift' in features:
            plot_feature_distribution(df, self.prep_file_name(
                self.name), 'redshift', self.class_labels, class_counts)

    def evaluate_model(self, roc_plots, class_metrics, acc_metrics, data_filters, class_counts, class_probabilities, X, y):
        """
        Evaluate and plot performance of model
        :param roc_plots: Mapping from class_name to [figure, axis, true positive rates, aucs]. This function plots curve on corresponding axis using this dict.
        :param class_metrics: Mapping from class names to SUMMED ranged metrics
        :param acc_metrics: List of results from get_mc_class_metrics, per fold/run
        :param data_filters: Map from data filter to value
        :param class_counts: Map from class labels to number of samples (from total y)
        :param class_probabilities: List of Numpy matrices returned from get_all_class_probabilities for each run
        :param X: all features in DataFrame
        :param y: DataFrame of TARGET_LABEL column for all data.
        """
        # Plot ROC curves for each class
        k = data_filters['folds']
        self.plot_mc_roc_curves(roc_plots, k)

        # Plot probability vs positive rate for each class
        self.plot_mc_probability_pos_rates(class_metrics)

        # self.plot_probability_dists(
        #     class_probabilities, data_filters['num_runs'], k, X, y)

        # Collect overall metrics
        agg_metrics = self.aggregate_mc_class_metrics(acc_metrics)

        print('debug: Aggregated metrics')
        print(agg_metrics)

        # Plot overall metrics
        self.plot_metrics(agg_metrics, class_counts, data_filters['prior'])

    def run_cross_validation(self, k, X, y, roc_plots, class_metrics, acc_metrics, cps, data_filters):
        """
        Run k-fold cross validation
        """
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            self.X_train, self.X_test = X.iloc[train_index].reset_index(
                drop=True), X.iloc[test_index].reset_index(drop=True)
            self.y_train, self.y_test = y.iloc[train_index].reset_index(
                drop=True), y.iloc[test_index].reset_index(drop=True)
            if data_filters['test_on_train']:
                self.X_test = self.X_train
                self.y_test = self.y_train

            # Apply PCA
            if data_filters['pca'] is not None:
                self.X_train, self.X_test = self.apply_pca(data_filters['pca'])

            # Run model
            self.train_model()

            probabilities = self.get_all_class_probabilities()
            # Save all probabilities for later analysis/plotting
            cps.append(probabilities)

            # Save ROC curve for each class
            roc_plots = self.save_roc_curve(roc_plots, probabilities)

            # Record metrics for prob vs. accuracy plots
            X_accs = self.get_mc_probability_matrix()

            for class_name in self.class_labels:
                class_metrics[class_name].append(self.get_mc_metrics_by_ranges(
                    X_accs, class_name))
            self.predictions = self.test_model()

            acc_metrics.append(self.get_mc_class_metrics())

        return roc_plots, class_metrics, acc_metrics, cps

    def run_model_cv(self, data_columns, data_filters):
        """
        Split data into k-folds and perform k-fold cross validation
        """
        k = data_filters['folds']
        X, y = get_source_target_data(data_columns, **data_filters)

        # 1. Add unspecified class labels to data
        y = self.add_unspecified_labels_to_data(y)

        # 2. Initialize self.class_labels based on data and add undefined classes
        self.set_class_labels(y, user_defined_labels=self.class_labels)

        # 3. Remove rows that don't have valid target label
        X, y = self.remove_excess_data(X, y)

        class_counts = self.get_class_counts(y)
        # 4. Initialize prob(parent |child) for inner nodes
        self.parent_probs = self.get_parent_probabilities(class_counts)

        # 5. Visualize data
        self.visualize_data(data_filters, y, X, class_counts)

        # 6. Run Cross-fold validation model.
        # Initialize maps of class names to metrics
        roc_plots = {}
        class_metrics = {}  # Probability vs precision
        for class_name in self.class_labels:
            roc_fig, roc_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            # roc_plots[class_name] = [fig  ax   TPRS  AUCS]
            roc_plots[class_name] = [roc_fig, roc_ax, [], []]
            class_metrics[class_name] = []

        acc_metrics = []  # List of maps from class to stats
        cps = []  # List of all class probabilities Numpy matrices
        for index_run in range(data_filters['num_runs']):
            print("\n\nRun " + str(index_run + 1))
            roc_plots, class_metrics, acc_metrics, cps = self.run_cross_validation(
                k, X, y, roc_plots, class_metrics, acc_metrics, cps, data_filters)

        # Plot Results ################################
        agg_class_metrics = self.aggregate_mc_prob_metrics(class_metrics)

        self.evaluate_model(roc_plots, agg_class_metrics,
                            acc_metrics, data_filters, class_counts, cps, X, y)

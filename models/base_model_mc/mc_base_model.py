from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

from thex_data.data_clean import *
from thex_data.data_plot import *
from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, UNDEF_CLASS, PRED_LABEL
from thex_data.data_prep import get_source_target_data
from thex_data.data_consts import class_to_subclass as hierarchy


from models.base_model.base_model import BaseModel
from models.base_model_mc.mc_base_model_performance import MCBaseModelPerformance
from models.base_model_mc.mc_base_model_plots import MCBaseModelVisualization


class MCBaseModel(BaseModel, MCBaseModelPerformance, MCBaseModelVisualization):
    """
    Abstract Class representing base functionality of all multiclass models. Inherits from BaseModel, and uses Mixins from other classes. Subclasses of models must implement all BaseModel abstract methods PLUS get_all_class_probabilities
    """

    def run_model(self):
        """
        Override run_model in BaseModel.

        Set custom attributes for Multiclass classifiers: test_level (level of class hierarchy over which to get probabilities, for MCKDEModel), and class_labels (specific classes to run model on)
        """
        for parent in hierarchy.keys():
            hierarchy[parent].append(UNDEF_CLASS + parent)
        self.tree = init_tree(hierarchy)
        self.class_levels = assign_levels(self.tree, {}, self.tree.root, 1)

        self.test_level = self.user_data_filters[
            'test_level'] if 'test_level' in self.user_data_filters.keys() else None

        self.class_labels = self.user_data_filters[
            'class_labels'] if 'class_labels' in self.user_data_filters.keys() else None

        super(MCBaseModel, self).run_model()

    def test_model(self, keep_top_half=False):
        """
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

    @abstractmethod
    def get_all_class_probabilities(self):
        """
        Get class probability for each sample, for each class.
        :return: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        pass

    def set_class_labels(self, y):

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

    def visualize_data(self, data_filters, y=None, X=None):
        """
        Visualize distribution of data used to train and test
        """
        if self.class_labels is None:
            self.set_class_labels(y)

        if y is None:
            y = pd.concat([self.y_train, self.y_test], axis=0)

        # Filter y to keep only rows with at least 1 class label in
        # self.class_labels
        keep_rows = []
        indices = []
        for df_index, row in y.iterrows():
            list_classes = convert_str_to_list(row[TARGET_LABEL])
            keep = False
            for c in list_classes:
                # Keep if at least one value in list is acceptable
                if c in self.class_labels or UNDEF_CLASS + c in self.class_labels:
                    keep = True
            if keep:
                keep_rows.append({TARGET_LABEL: row[TARGET_LABEL]})
                indices.append(df_index)
        self.y_filtered = pd.DataFrame(keep_rows).reset_index(drop=True)

        plot_class_hist(self.y_filtered, True)

        if X is not None:
            # pass in X and y combined
            X_filtered = X.iloc[indices].reset_index(drop=True)
            df = pd.concat([X_filtered, self.y_filtered], axis=1)
            features = list(df)
            if 'redshift' in features:
                plot_feature_distribution(df, 'redshift', False)

    def evaluate_model(self, roc_plots, class_metrics, acc_metrics, k):
        """
        Evaluate and plot performance of model
        :param roc_plots: Mapping from class_name to [figure, axis, true positive rates, aucs]. This function plots curve on corresponding axis using this dict.
        :param class_metrics: Mapping from class names to SUMMED ranged metrics
        :param acc_metrcis: List of results from get_mc_class_metrics, per fold/run
        :param k: Number of folds
        """
        # Plot ROC curves for each class
        self.plot_mc_roc_curves(roc_plots, k)

        # Plot probability vs positive rate for each class
        self.plot_mc_probability_pos_rates(class_metrics)

        # Report overall metrics
        agg_metrics = self.aggregate_mc_class_metrics(acc_metrics)
        precisions = {}
        recalls = {}
        briers = {}
        loglosses = {}
        for class_name in self.class_labels:
            metrics = agg_metrics[class_name]
            precisions[class_name] = metrics["TP"] / (metrics["TP"] + metrics["FP"])
            recalls[class_name] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
            briers[class_name] = metrics["BS"]
            loglosses[class_name] = metrics["LL"]
        self.plot_mc_performance(precisions, "Precision")
        self.plot_mc_performance(recalls, "Recall")
        # self.basic_plot(briers, "Brier Score",   self.class_labels)
        self.basic_plot(loglosses,  "Neg Log Loss",  self.class_labels)

    def run_cross_validation(self, k, X, y, roc_plots, class_metrics, acc_metrics, data_filters):
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

            # Save ROC curve for each class
            roc_plots = self.save_roc_curve(roc_plots)

            # Record metrics for prob vs. accuracy plots
            X_accs = self.get_mc_probability_matrix()

            for class_name in self.class_labels:
                class_metrics[class_name].append(self.get_mc_metrics_by_ranges(
                    X_accs, class_name))
            self.predictions = self.test_model()

            acc_metrics.append(self.get_mc_class_metrics())

        return roc_plots, class_metrics, acc_metrics

    def run_model_cv(self, data_columns, data_filters):
        """
        Split data into k-folds and perform k-fold cross validation
        """
        k = data_filters['folds']
        X, y = get_source_target_data(data_columns, **data_filters)
        # Initialize self.class_labels if None
        if self.class_labels is None:
            self.set_class_labels(y)
        self.visualize_data(data_filters, y, X)

        # Initialize maps of class names to metrics
        roc_plots = {}
        class_metrics = {}
        for class_name in self.class_labels:
            roc_fig, roc_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            # roc_plots[class_name] = [fig  ax   TPRS  AUCS]
            roc_plots[class_name] = [roc_fig, roc_ax, [], []]
            class_metrics[class_name] = []

        acc_metrics = []  # List of maps from class to stats
        for index_run in range(data_filters['num_runs']):
            print("\n\nRun " + str(index_run + 1))
            roc_plots, class_metrics, acc_metrics = self.run_cross_validation(
                k, X, y, roc_plots, class_metrics, acc_metrics, data_filters)

        # Plot Results ################################
        agg_class_metrics = self.aggregate_mc_prob_metrics(class_metrics)
        self.evaluate_model(roc_plots, agg_class_metrics, acc_metrics, k)

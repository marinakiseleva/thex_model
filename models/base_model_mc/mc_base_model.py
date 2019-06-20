import os
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from thex_data.data_clean import *
from thex_data.data_consts import ROOT_DIR, FIG_WIDTH, FIG_HEIGHT, DPI, UNDEF_CLASS, PRED_LABEL
from thex_data.data_prep import get_source_target_data
from thex_data.data_plot import *
from thex_data.data_consts import class_to_subclass as hierarchy


from models.base_model.base_model import BaseModel
from models.base_model_mc.mc_base_model_performance import MCBaseModelPerformance
from models.base_model_mc.mc_base_model_plots import MCBaseModelVisualization


class MCBaseModel(BaseModel, MCBaseModelPerformance, MCBaseModelVisualization):
    """
    Abstract Class representing base functionality of all multiclass models. Inherits from BaseModel, and uses Mixins from other classes. Subclasses of models must implement all BaseModel methods PLUS test_probabilities
    """

    def test_model(self, keep_top_half=True):
        """
        Get class prediction for each sample. Predict class with max probability density.
        :return: DataFrame with column PRED_LABEL, which contains the class name of the class with the highest probability assigned.
        """
        # For diagnosing purposes - keep only top 1/2 probs
        if keep_top_half:
            unnormalized_max_probabilities = []
            for index, row in self.X_test.iterrows():
                # TODO: how to change if probabilities are normalized or unnormalized.
                unnormalized_probabilities = self.get_class_probabilities(
                    row)
                max_unnormalized_prob = max(unnormalized_probabilities.values())
                unnormalized_max_probabilities.append(max_unnormalized_prob)
            probs = np.array(unnormalized_max_probabilities)
            keep_indices = np.argwhere(probs > np.average(probs)).transpose()[0].tolist()

            # Get max class for those indices, and filter down self.X_test and
            # self.y_test to have same rows
            self.X_test = self.X_test.loc[self.X_test.index.isin(keep_indices)]
            self.y_test = self.y_test.loc[self.y_test.index.isin(keep_indices)]

        predictions = []
        for index, row in self.X_test.iterrows():
            probabilities = self.get_class_probabilities(row)
            max_prob_class = max(probabilities, key=probabilities.get)
            predictions.append(max_prob_class)
        predicted_classes = pd.DataFrame(predictions, columns=[PRED_LABEL])
        return predicted_classes

    def run_model(self):
        """
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

    @abstractmethod
    def get_all_class_probabilities(self):
        """
        Get class probability for each sample, for each class.
        :return: Numpy Matrix with each row corresponding to sample, and each column the probability of that class
        """
        pass

    def evaluate_model(self, test_on_train):
        """
        Evaluate and plot performance of model
        """
        # class_recalls, class_precisions = self.get_mc_metrics()
        self.plot_mc_roc_curves()

        # Plot probability vs precision for each class
        X_accs = self.get_mc_probability_matrix()
        for class_name in self.class_labels:
            perc_ranges, AP, TOTAL = self.get_mc_metrics_by_ranges(
                X_accs, class_name)
            acc_ranges = [AP[index] / T if T > 0 else 0 for index, T in enumerate(TOTAL)]
            self.plot_probability_ranges(perc_ranges, acc_ranges,
                                         'TP/Total', class_name, TOTAL)

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

    def run_cross_validation(self, k, X, y, roc_plots, class_metrics, acc_metrics, data_filters):
        """
        Run k-fold cross validation
        """
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        i = 1  # Fold count, for plotting labels
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
            # self.test_model()

            # Save ROC curve for each class
            roc_plots = self.save_roc_curve(i, roc_plots)
            i += 1

            # Record metrics for prob vs. accuracy plots
            X_accs = self.get_mc_probability_matrix()
            for class_name in self.class_labels:
                class_metrics[class_name].append(self.get_mc_metrics_by_ranges(
                    X_accs, class_name))
            self.predictions = self.test_model()
            if self.test_level is not None:
                acc_metrics.append(self.get_mc_class_metrics())

        return roc_plots, class_metrics, acc_metrics

    def set_class_labels(self, y):
        if self.test_level is not None:
            # Gets all class labels on self.test_level,
            # including 'Undefined' classes of parent level.
            self.class_labels = []
            for class_name in self.get_mc_unique_classes(y):
                class_level = self.class_levels[class_name]
                if class_level == self.test_level:
                    self.class_labels.append(class_name)
                elif class_level == self.test_level - 1:
                    self.class_labels.append(UNDEF_CLASS + class_name)
        else:
            self.class_labels = self.get_mc_unique_classes(y)

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
        # Plot ROC curves for each class
        avg_fig, avg_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        for class_name in self.class_labels:
            fig, ax, tprs, aucs = roc_plots[class_name]
            #   Baseline
            ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='r',
                    label='Baseline', alpha=.8)
            # Average line
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_fpr = np.linspace(0, 1, 100)
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC=%0.2f$\pm$%0.2f)' % (
                        mean_auc, std_auc),
                    lw=2, alpha=.8)
            avg_ax.plot(mean_fpr, mean_tpr, lw=1, alpha=0.6,
                        label=class_name + r' AUC=%0.2f$\pm$%0.2f' % (
                            mean_auc, std_auc))
            # Standard deviation in gray shading
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\sigma$')
            title = class_name + " ROC Curve " + "over " + str(k) + "-folds"
            ax.set_title(title)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="best")
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            file_dir = ROOT_DIR + "/output/" + self.prep_file_name(self.name)
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
            file_name = file_dir + "/" + self.prep_file_name(title)
            fig.savefig(file_name, bbox_inches=extent.expanded(1.3, 1.3))

        avg_ax.set_title(self.name + " ROC Curves")
        avg_ax.set_xlabel('False Positive Rate')
        avg_ax.set_ylabel('True Positive Rate')
        avg_ax.legend(loc="best")
        avg_fig.savefig(file_dir + "/roc_summary",
                        bbox_inches=extent.expanded(1.3, 1.3))
        plt.show()

        # Plot probability vs accuracy for each class
        aggregated_class_metrics = self.aggregate_mc_metrics(class_metrics)
        self.plot_mc_probability_precision(aggregated_class_metrics)

        # Report overall metrics: precision, recall, accuracy
        if self.test_level is not None:
            agg_metrics = self.aggregate_mc_class_metrics(acc_metrics)
            precisions = {}
            recalls = {}
            accuracy = 0
            for class_name in self.class_labels:
                [TP, FN, FP, TN] = agg_metrics[class_name]
                precisions[class_name] = TP / (TP + FP)
                recalls[class_name] = TP / (TP + FN)
                accuracy += TP
            print("total tn " + str(accuracy))
            total = data_filters['num_runs'] * self.y_test.shape[0]
            print("Total number " + str(total))
            accuracy = accuracy / total
            print("Overall accuracy " + str(round(accuracy, 2) * 100) + "%")
            self.plot_performance(precisions, "Precision", class_counts=None,
                                  ylabel="Precision", class_names=self.class_labels)
            self.plot_performance(recalls, "Recall", class_counts=None,
                                  ylabel="Recall", class_names=self.class_labels)

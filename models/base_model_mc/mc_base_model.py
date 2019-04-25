from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from thex_data import data_plot
from thex_data.data_prep import get_source_target_data
from thex_data.data_print import *

from models.base_model.base_model import BaseModel
from models.base_model_mc.mc_base_model_performance import MCBaseModelPerformance
from models.base_model_mc.mc_base_model_plots import MCBaseModelVisualization


FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 300


class MCBaseModel(BaseModel, MCBaseModelPerformance, MCBaseModelVisualization):
    """
    Abstract Class representing base functionality of all multiclass models. Inherits from BaseModel, and uses Mixins from other classes. Subclasses of models must implement all BaseModel methods PLUS test_probabilities
    """

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
        class_recalls, class_precisions = self.get_mc_metrics()
        # self.plot_performance(class_recalls, self.name + " Recall",
        #                       class_counts=None, ylabel="Recall")
        # self.plot_performance(class_precisions, self.name + " Precision",
        #                       class_counts=None, ylabel="Precision")

        self.plot_mc_roc_curves(self.models)

        # Plot probability vs precision for each class
        X_accs = self.get_mc_probability_matrix()
        unique_classes = self.get_mc_unique_classes()
        for class_index, class_name in enumerate(unique_classes):
            perc_ranges, AP, TOTAL = self.get_mc_metrics_by_ranges(
                X_accs, class_name)
            acc = [AP[index] / T if T > 0 else 0 for index, T in enumerate(TOTAL)]
            self.plot_probability_ranges(perc_ranges, acc,
                                         'TP/Total', class_name, TOTAL)

    def run_cross_validation(self, k, X, y, data_filters):
        """
        Run k-fold cross validation
        """
        kf = StratifiedKFold(n_splits=k, shuffle=True)

        model_classes = self.get_mc_unique_classes(df=y)

        roc_plots = {}
        for class_name in model_classes:
            roc_fig, roc_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
            # roc_plots[class_name] = [fig  ax   TPRS  AUCS]
            roc_plots[class_name] = [roc_fig, roc_ax, [], []]

        i = 1  # Fold count, for plotting labels
        for train_index, test_index in kf.split(X, y):
            print("first fold")
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
            roc_plots = self.save_roc_curve(i, roc_plots, model_classes)
            i += 1
        return roc_plots

    def run_model_cv(self, data_columns, data_filters):
        """
        Split data into k-folds and perform k-fold cross validation
        """
        k = data_filters['folds']
        X, y = get_source_target_data(data_columns, **data_filters)

        # Initialize metric collections over all runs
        class_metrics = []
        for index_run in range(data_filters['num_runs']):
            roc_plots = self.run_cross_validation(k, X, y, data_filters)

        for class_name in roc_plots.keys():
            f, ax, tprs, aucs = roc_plots[class_name]
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
            # Standard deviation in gray shading
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\sigma$')

            ax.set_title(class_name + " ROC Curve " + " over " + str(k) + "-folds")
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="best")

        plt.show()

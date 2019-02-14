import argparse
from thex_data.data_init import *
from thex_data.data_consts import drop_cols
from thex_data.data_prep import get_train_test

from abc import ABC, abstractmethod

from models.nb_model.nb_performance import get_rocs
from model_performance.performance import compute_plot_class_accuracy, plot_confusion_matrix, get_accuracy
from thex_data import data_plot


class BaseModel(ABC):
    """
    Abstract Class representing base functionality of all models. Subclasses of models implement their own training and testing functions. 
    """

    def run_model(self, cols=None, col_match=None, test_on_train=False,  folds=3, **data_filters):
        """
        Collects data based on column parameters, trains, and tests model. Either cols or col_match need to be passed in, otherwise all numeric columns are used.
        :param cols: Names of columns to filter on
        :param col_match: String by which columns will be selected. For example: AllWISE will use all AlLWISE columns.
        :param test_on_train: Boolean to test on training data.
        :param folds: Number of folds to use in k-fold Cross Validation
        """

        col_list = []
        if cols is not None:
            col_list = cols
        else:
            if col_match is not None:
                col_list = collect_cols(col_match)
            else:
                # Use all columns in data set with numeric values
                cols = list(collect_data())
                col_list = []
                for c in cols:
                    if not any(drop in c for drop in drop_cols):
                        col_list.append(c)  # Keep only numeric columns
        # Collect data filtered on these parameters
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_model_data(
            col_list, test_on_train, **data_filters)
        self.visualize_data()
        self.train_model()
        predictions = self.test_model()
        self.evaluate_model(predictions)

    def visualize_data(self):
        """
        Visualize distribution of data used to train and test
        """
        labels = pd.concat([self.y_train, self.y_test], axis=0)
        data_plot.plot_class_hist(labels)

    def evaluate_model(self, predicted_classes):
        """
        Evaluate and plot performance of model
        """

        total_accuracy = get_accuracy(predicted_classes, self.y_test)
        model_name = self.name
        compute_plot_class_accuracy(
            predicted_classes, self.y_test, plot_title=model_name + " Accuracy, on Testing Data")

        # get_rocs(X_test, y_test, summaries, priors)

        # for f in list(X_train):
        #     data_plot.plot_feature_distribution(concat([X_train, y_train], axis=1), f)

        plot_confusion_matrix(self.y_test, predicted_classes, normalize=True)

    def run_cv(data_columns, incl_redshift=False, test_on_train=False, k=3):
        """
        Split data into k-folds and perform k-fold cross validation
        """
        X, y = get_source_target_data(data_columns, incl_redshift)
        kf = KFold(n_splits=k, shuffle=True, random_state=10)
        nb_recall, tree_recall = [], []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            data_type = ' Testing '
            if test_on_train:  # Update fields to test models on training data
                X_test = X_train.copy()
                y_test = y_train.copy()
                data_type = ' Training '

            nb_predictions, tree_predictions = run_models(
                X_train, y_train, X_test, y_test)

            # Save Naive Bayes accuracy
            nb_class_accuracies = get_class_accuracies(
                combine_dfs(nb_predictions, y_test))
            nb_recall.append(nb_class_accuracies)

            # Save tree accuracy
            tree_class_accuracies = get_class_accuracies(
                combine_dfs(tree_predictions, y_test))
            tree_recall.append(tree_class_accuracies)

        # Get average accuracy per class (mapping)
        avg_tree_acc = aggregate_accuracies(tree_recall, k, y)
        avg_nb_acc = aggregate_accuracies(nb_recall, k, y)
        plot_class_accuracy(
            avg_tree_acc, plot_title="HSC Tree: " + str(k) + "-fold CV on" + data_type + " data")
        plot_class_accuracy(
            avg_nb_acc, plot_title="Naive Bayes: " + str(k) + "-fold CV on" + data_type + " data")

    def get_model_data(self, col_list, test_on_train, **data_filters):
        X_train, X_test, y_train, y_test = get_train_test(
            col_list, **data_filters)
        if test_on_train == True:
            X_test = X_train.copy()
            y_test = y_train.copy()
        return X_train, X_test, y_train, y_test

    @abstractmethod
    def train_model(self):
        """
        Train model using training data, self.X_train self.y_train
        """
        pass

    @abstractmethod
    def test_model(self):
        """
        Test model using testing data self.X_test self.y_test
        Returns predictions (in class code format)
        """
        pass

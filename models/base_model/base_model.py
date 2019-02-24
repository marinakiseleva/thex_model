from abc import ABC, abstractmethod
from sklearn.model_selection import KFold

from thex_data.data_init import *
from thex_data.data_consts import drop_cols
from thex_data.data_prep import get_train_test, get_source_target_data
from thex_data.data_print import print_filters
from thex_data import data_plot
from models.base_model.base_model_performance import BaseModelPerformance
from models.base_model.base_model_plots import BaseModelVisualization


class BaseModel(ABC, BaseModelPerformance, BaseModelVisualization):
    """
    Abstract Class representing base functionality of all models. Subclasses of models implement their own training and testing functions. 
    """

    def run_model(self, cols=None, col_matches=None, folds=None, **user_data_filters):
        """
        Collects data based on column parameters, trains, and tests model. Either cols or col_matches need to be passed in, otherwise all numeric columns are used.
        :param cols: Names of columns to filter on
        :param col_matches: String by which columns will be selected. For example: AllWISE will use all AlLWISE columns.
        :param folds: Number of folds if using k-fold Cross Validation
        :param user_data_filters: List of data filters user passed in. User options over-ride any default.
        """
        # Set defaults on filters
        data_filters = {'num_runs': 1,
                        'test_on_train': False,
                        'top_classes': 10,
                        'one_all': None,
                        'data_split': 0.3,
                        'subsample': 200,
                        'transform_features': False,
                        'incl_redshift': False

                        }
        # Update filters with any passed-in filters
        for data_filter in user_data_filters.keys():
            data_filters[data_filter] = user_data_filters[data_filter]
        print_filters(data_filters)

        col_list = collect_cols(cols, col_matches)

        if isinstance(folds, int):
            self.run_model_cv(col_list, folds, data_filters)
            return 0

        # Collect data filtered on these parameters
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_model_data(
            col_list, **data_filters)
        self.visualize_data()

        self.train_model()
        self.predictions = self.test_model()
        self.evaluate_model(data_filters['test_on_train'])
        return self

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

    def get_model_data(self, col_list, **data_filters):
        """
        Collects data for model
        :param col_list: List of columns to filter on
        :param test_on_train: Boolean to test on training data.
        :param user_data_filters: List of data filters user passed in.
        """
        X_train, X_test, y_train, y_test = get_train_test(
            col_list, **data_filters)
        if data_filters['test_on_train'] == True:
            X_test = X_train.copy()
            y_test = y_train.copy()
        return X_train, X_test, y_train, y_test

    def visualize_data(self, y=None):
        """
        Visualize distribution of data used to train and test
        """
        if y is None:
            labels = pd.concat([self.y_train, self.y_test], axis=0)
        else:
            labels = y
        data_plot.plot_class_hist(labels)

    def evaluate_model(self, test_on_train):
        """
        Evaluate and plot performance of model
        """

        self.plot_roc_curves()
        # Get accuracy per class of transient
        class_accuracies = self.get_class_accuracies()
        class_counts = self.get_class_counts(class_accuracies.keys())
        data_type = 'Training' if test_on_train else 'Testing'
        title = self.name + " Accuracy on " + data_type + " Data"
        self.plot_class_accuracy(title, class_accuracies, class_counts)
        self.plot_confusion_matrix(normalize=True)

    def run_cross_validation(self, k, X, y, data_filters):
        """
        Run k-fold cross validation
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=10)
        accuracies = []
        for train_index, test_index in kf.split(X):
            self.X_train, self.X_test = X.iloc[train_index].reset_index(
                drop=True), X.iloc[test_index].reset_index(drop=True)
            self.y_train, self.y_test = y.iloc[train_index].reset_index(
                drop=True), y.iloc[test_index].reset_index(drop=True)

            if data_filters['test_on_train']:
                self.X_test = self.X_train
                self.y_test = self.y_train

            # Save model accuracy
            self.train_model()
            self.predictions = self.test_model()
            accuracies.append(self.get_class_accuracies())

        # Get average accuracy per class (mapping)
        avg_acc = self.aggregate_accuracies(accuracies, y)
        return avg_acc

    def run_model_cv(self, data_columns, k, data_filters):
        """
        Split data into k-folds and perform k-fold cross validation
        """
        X, y = get_source_target_data(data_columns, **data_filters)
        self.visualize_data(y)
        run_accuracies = []  # list of model accuracies from each run
        for index_run in range(data_filters['num_runs']):
            run_accuracies.append(self.run_cross_validation(k, X, y, data_filters))
        avg_acc = self.aggregate_accuracies(run_accuracies, y)
        data_type = 'Training' if data_filters['test_on_train'] else 'Testing'
        title = self.name + " Average Accuracy from " + \
            str(data_filters['num_runs']) + " runs of " + \
            str(k) + "-fold CV on " + data_type + " data"
        self.plot_class_accuracy(title, avg_acc)

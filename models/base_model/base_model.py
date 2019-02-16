from abc import ABC, abstractmethod
from sklearn.model_selection import KFold

from thex_data.data_init import *
from thex_data.data_consts import drop_cols
from thex_data.data_prep import get_train_test, get_source_target_data
from thex_data.data_print import print_filters
from thex_data import data_plot
from models.base_model.base_model_performance import *


class BaseModel(ABC):
    """
    Abstract Class representing base functionality of all models. Subclasses of models implement their own training and testing functions. 
    """

    def run_model(self, cols=None, col_matches=None, test_on_train=False,  folds=None, **user_data_filters):
        """
        Collects data based on column parameters, trains, and tests model. Either cols or col_matches need to be passed in, otherwise all numeric columns are used.
        :param cols: Names of columns to filter on
        :param col_matches: String by which columns will be selected. For example: AllWISE will use all AlLWISE columns.
        :param test_on_train: Boolean to test on training data.
        :param folds: Number of folds to use in k-fold Cross Validation
        :param user_data_filters: List of data filters user passed in. User options over-ride any default.
        """
        # Set defaults on filters
        data_filters = {'top_classes': 10, 'one_all': None, 'data_split': 0.3,
                        'subsample': 200, 'transform_features': False, 'incl_redshift': False}
        # Update filters with any passed-in filters
        for data_filter in user_data_filters.keys():
            data_filters[data_filter] = user_data_filters[data_filter]
        print_filters(data_filters)

        col_list = collect_cols(cols, col_matches)
        if isinstance(folds, int):
            self.run_model_cv(col_list, folds, test_on_train, data_filters)
            return 0

        # Collect data filtered on these parameters
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_model_data(
            col_list, test_on_train, **data_filters)
        self.visualize_data()
        self.train_model()
        predictions = self.test_model()
        self.evaluate_model(predictions)
        return self

    def get_model_data(self, col_list, test_on_train, **data_filters):
        """
        Collects data for model
        :param col_list: List of columns to filter on
        :param test_on_train: Boolean to test on training data.
        :param user_data_filters: List of data filters user passed in.
        """
        X_train, X_test, y_train, y_test = get_train_test(
            col_list, **data_filters)
        if test_on_train == True:
            X_test = X_train.copy()
            y_test = y_train.copy()
        return X_train, X_test, y_train, y_test

    def visualize_data(self):
        """
        Visualize distribution of data used to train and test
        """
        labels = pd.concat([self.y_train, self.y_test], axis=0)
        data_plot.plot_class_hist(labels)

    def evaluate_model(self, predicted_classes):
        """
        Evaluate and plot performance of model
        :param predicted_classes: classes predicted by label (as class codes)
        """
        total_accuracy = get_accuracy(predicted_classes, self.y_test)
        model_name = self.name
        compute_plot_class_accuracy(
            predicted_classes, self.y_test, plot_title=model_name + " Accuracy, on Testing Data")

        # for f in list(X_train):
        #     data_plot.plot_feature_distribution(concat([X_train, y_train], axis=1), f)

        plot_confusion_matrix(self.y_test, predicted_classes, normalize=True)

    def run_model_cv(self, data_columns, k, test_on_train, data_filters):
        """
        Split data into k-folds and perform k-fold cross validation
        """
        X, y = get_source_target_data(data_columns, **data_filters)
        kf = KFold(n_splits=k, shuffle=True, random_state=10)
        accuracies = []
        for train_index, test_index in kf.split(X):
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]

            if test_on_train:
                self.X_test = self.X_train
                self.y_test = self.y_train

            # Save model accuracy
            self.train_model()
            predictions = self.test_model()
            accuracies.append(get_class_accuracies(
                combine_dfs(predictions, self.y_test)))

        # Get average accuracy per class (mapping)
        avg_acc = aggregate_accuracies(accuracies, k, y)
        data_type = 'Training' if test_on_train else 'Testing'
        plot_class_accuracy(
            avg_acc, plot_title=self.name + " " + str(k) + "-fold CV on " + data_type + " data")

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

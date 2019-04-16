from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, StratifiedKFold

from thex_data.data_init import *
from thex_data.data_consts import drop_cols
from thex_data.data_prep import get_train_test, get_source_target_data
from thex_data.data_print import *
from thex_data import data_plot
from models.base_model.base_model_performance import BaseModelPerformance
from models.base_model.base_model_plots import BaseModelVisualization

from sklearn.decomposition import PCA


class BaseModel(ABC, BaseModelPerformance, BaseModelVisualization):
    """
    Abstract Class representing base functionality of all models. Subclasses of models implement their own training and testing functions.
    """

    def run_model(self):
        """
        Collects data based on column parameters, trains, and tests model. Either cols or col_matches need to be passed in, otherwise all numeric columns are used.
        :param cols: Names of columns to filter on
        :param col_matches: String by which columns will be selected. For example: AllWISE will use all AllWISE columns.
        :param folds: Number of folds if using k-fold Cross Validation
        :param user_data_filters: List of data filters user passed in. User options over-ride any default.
        """
        cols = self.cols
        col_matches = self.col_matches
        user_data_filters = self.user_data_filters

        # Set defaults on filters
        data_filters = {'num_runs': 1,
                        'test_on_train': False,
                        'folds': None,
                        'data_split': 0.3,  # For single run
                        'hierarchical_model': False,
                        'top_classes': None,
                        'one_all': None,
                        'subsample': None,
                        'transform_features': False,
                        'transform_labels': True,
                        'incl_redshift': False,
                        'pca': None  # Number of principal components

                        }
        # Update filters with any passed-in filters
        for data_filter in user_data_filters.keys():
            data_filters[data_filter] = user_data_filters[data_filter]
        print_filters(data_filters)

        col_list = collect_cols(cols, col_matches)
        print_features_used(col_list)
        if data_filters['folds'] is not None:
            self.run_model_cv(col_list,  data_filters)
            return 0

        # Collect data filtered on these parameters
        self.set_model_data(col_list, **data_filters)
        self.visualize_data()
        self.train_model()
        self.predictions = self.test_model()
        if data_filters['hierarchical_model'] == False:
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

    @abstractmethod
    def get_class_probabilities(self, x):
        """
        Get probability per class for this test point x
        """
        pass

    def set_model_data(self, col_list, **data_filters):
        """
        Collects data for model
        :param col_list: List of columns to filter on
        :param test_on_train: Boolean to test on training data.
        :param user_data_filters: List of data filters user passed in.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = get_train_test(
            col_list, **data_filters)
        if data_filters['test_on_train'] == True:
            self.X_test = X_train.copy()
            self.y_test = y_train.copy()
        # Apply PCA
        if data_filters['pca'] is not None:
            self.X_train, self.X_test = self.apply_pca(data_filters['pca'])

    def apply_pca(self, k):
        """
        Compute PCA reductions on training then apply to both training and testing.
        """
        pca = PCA(n_components=k)
        pca = pca.fit(self.X_train)  # Fit on training
        reduced_training = pca.transform(self.X_train)
        reduced_testing = pca.transform(self.X_test)
        # print("\nPCA Analysis: Explained Variance Ratio")
        # print(pca.explained_variance_ratio_)

        def convert_to_df(data, k):
            reduced_columns = []
            for i in range(k):
                reduced_columns.append("PC" + str(i + 1))
            return pd.DataFrame(data=data, columns=reduced_columns)

        reduced_training = convert_to_df(reduced_training, k)
        reduced_testing = convert_to_df(reduced_testing, k)
        return reduced_training, reduced_testing

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
        self.plot_probability_correctness()
        self.plot_roc_curves()
        # Get accuracy per class of transient
        class_recalls = self.get_class_recalls()
        class_precisions = self.get_class_precisions()
        class_counts = self.get_class_counts(class_recalls.keys())
        data_type = 'Training' if test_on_train else 'Testing'
        title = self.name + " Recall on " + data_type + " Data"
        self.plot_accuracies(class_recalls, title, class_counts, ylabel="Recall")

        class_counts = self.get_class_counts(class_precisions.keys())
        title = self.name + " Precision on " + data_type + " Data"
        self.plot_accuracies(class_precisions, title, class_counts, ylabel="Precision")
        self.plot_confusion_matrix(normalize=True)

    def run_cross_validation(self, k, X, y, data_filters):
        """
        Run k-fold cross validation
        """
        kf = StratifiedKFold(n_splits=k, shuffle=True)
        recalls = []
        precisions = []
        unique_classes = self.get_unique_classes(y)
        prob_ranges = {class_code: [] for class_code in unique_classes}
        class_rocs = {class_code: [] for class_code in unique_classes}
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
            self.predictions = self.test_model()

            # Save model accuracy
            X_preds = pd.concat(
                [self.get_probability_matrix(), self.test_model()], axis=1)
            for class_code in unique_classes:
                # Record ROC performance
                FP_rates, TP_rates = self.get_roc_curve(class_code)
                class_rocs[class_code].append([FP_rates, TP_rates])
                # Record probability vs accuracy
                perc_ranges, corr_ranges, count_ranges = self.get_corr_prob_ranges(
                    X_preds, class_code)
                prob_ranges[class_code].append([perc_ranges, corr_ranges, count_ranges])
            recalls.append(self.get_class_recalls())
            precisions.append(self.get_class_precisions())

        # Get average accuracy per class (mapping)
        avg_recalls = self.aggregate_accuracies(recalls, unique_classes)
        avg_precisions = self.aggregate_accuracies(precisions, unique_classes)
        avg_rocs = self.aggregate_rocs(class_rocs)
        avg_prob_ranges = self.aggregate_prob_ranges(prob_ranges)
        return avg_recalls, avg_precisions, avg_rocs, avg_prob_ranges

    def run_model_cv(self, data_columns, data_filters):
        """
        Split data into k-folds and perform k-fold cross validation
        """
        k = data_filters['folds']
        X, y = get_source_target_data(data_columns, **data_filters)
        if data_filters['transform_labels']:
            self.visualize_data(y)

        # Initialize metric collections over all runs
        class_recalls = []  # list of model accuracies/class recalls from each run
        class_precisions = []  # precision of each class from each run
        unique_classes = self.get_unique_classes(y)
        # true positive/false positive rates per class from each run
        all_rocs = {class_code: [] for class_code in unique_classes}
        all_prob_ranges = {class_code: [] for class_code in unique_classes}
        for index_run in range(data_filters['num_runs']):
            cur_recalls, cur_precisions, cur_rocs, cur_probs = self.run_cross_validation(
                k, X, y, data_filters)
            class_recalls.append(cur_recalls)
            class_precisions.append(cur_precisions)
            for class_code in unique_classes:
                all_rocs[class_code].append(cur_rocs[class_code])
                all_prob_ranges[class_code].append(cur_probs[class_code])

        avg_recalls = self.aggregate_accuracies(class_recalls, unique_classes)
        avg_precisions = self.aggregate_accuracies(class_precisions, unique_classes)
        avg_rocs = self.aggregate_rocs(all_rocs)
        avg_prob_ranges = self.aggregate_prob_ranges(all_prob_ranges)

        data_type = 'Training' if data_filters['test_on_train'] else 'Testing'
        info = " of " + self.name + " from " + str(data_filters['num_runs']) + " runs of " + \
            str(k) + "-fold CV on " + data_type + " data"

        self.plot_accuracies(avg_recalls, "Recall" + info, ylabel="Recall")
        self.plot_accuracies(avg_precisions, "Precision" + info, ylabel="Precision")
        self.plot_roc_curves(avg_rocs, "ROCs" + info)
        self.plot_probability_correctness(
            avg_prob_ranges, "Accuracy vs Probability" + info)

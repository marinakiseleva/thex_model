"""
data_prep
Prepares data before running through a classifier. Contains functionality to pull down data, filter data, enhances features, and split into training and testing sets.
"""
import sys
from sklearn.model_selection import train_test_split

from .data_clean import *
from .data_consts import TARGET_LABEL
from .data_filter import *
from .data_init import collect_data
from .data_plot import *
from .data_transform import transform_features
from .data_print import *


def get_data(col_list, **data_filters):
    """
    Pull in data and filter based on different biases and corrections: group transient types, fitler to passed-in columns, keep only rows with at least 1 valid value, filter to most frequent classes, sub-sample each class to same number, and take difference between wavelengths to make new features
    :param data_columns: List of columns to filter data on
    :param **data_filters: Mapping of data filters to values passed in by user
    """

    df = collect_data()
    df = group_by_tree(df, data_filters['transform_labels'])
    df = filter_columns(df.copy(), col_list, data_filters['incl_redshift'])

    df.dropna(axis=0, inplace=True)

    # Plots Entire Feature Dist
    if data_filters['transform_labels']:
        plot_feature_distribution(df, "redshift")

    # Keep only some classes, and turn remaining to 'Other'
    df = one_all(df, data_filters['one_all'])

    # Filter to most popular classes
    df = filter_top_classes(df, data_filters['top_classes'])

    # Randomly subsample any over-represented classes down to 100
    df = sub_sample(df, count=data_filters['subsample'])

    if df.shape[0] == 0:
        print("\n\nNo data to run model on. Try changing data filters or limiting number of features. Note: Running on all columns will not work since no data spans all features.\n\n")
        sys.exit()

    if data_filters['transform_labels']:
        print_class_counts(df)

    return df.reset_index(drop=True)


def get_source_target_data(data_columns, **data_filters):
    """
    Gets data split into source and target; but not yet split into training and testing
    """
    data = get_data(col_list=data_columns, **data_filters)
    X = data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
    if data_filters['transform_features']:
        X = transform_features(X)

    if data_filters['transform_labels']:
        y = data[[TARGET_LABEL]].astype(int).reset_index(drop=True)
    else:
        y = data[[TARGET_LABEL]]

    return X, y


def get_train_test(data_columns, **data_filters):
    """
    Using data filters passed in to get data, and returns data split into features and labels for training and testing
    :param data_columns: List of columns to filter data on
    :param **data_filters: Mapping of data filters to values passed in by user
    """
    X, y = get_source_target_data(data_columns, **data_filters)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_filters['data_split'])
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)

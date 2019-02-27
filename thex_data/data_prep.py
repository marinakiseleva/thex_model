"""
Connection between data manipulation and models: calls upon other functionalities that pull down data, filters it, and enhances features.
"""
import sys
from sklearn.model_selection import train_test_split

from .data_transform import transform_features
from .data_filter import *
from .data_consts import TARGET_LABEL
from .data_print import *
from .data_init import collect_data
from .data_clean import *
from . import data_plot

def get_data(col_list, **data_filters):
    """
    Pull in data and filter based on different biases and corrections: group transient types, fitler to passed-in columns, keep only rows with at least 1 valid value, filter to most frequent classes, sub-sample each class to same number, and take difference between wavelengths to make new features
    :param data_columns: List of columns to filter data on
    :param **data_filters: Mapping of data filters to values passed in by user
    """

    df = collect_data()
    df = group_cts(df)
    df = filter_columns(df.copy(), col_list, data_filters['incl_redshift'])
    df.dropna(axis=0, inplace=True)

    # Filter to most popular classes
    df = filter_top_classes(df, top=data_filters['top_classes'])

    # Keep only some classes, and turn remaining to 'Other'
    df = one_all(df, data_filters['one_all'])

    # Randomly subsample any over-represented classes down to 100
    df = sub_sample(df, count=data_filters['subsample'], col_val=TARGET_LABEL)

    if df.shape[0] == 0:
        print("\n\nNo data to run model on. Try changing data filters or limiting number of features. Note: Running on all columns will not work since no data spans all features.\n\n")
        sys.exit()

    print_class_counts(df)
    return df.reset_index(drop=True)


def get_source_target_data(data_columns, **data_filters):
    data = get_data(col_list=data_columns, **data_filters)
    X = data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
    if data_filters['transform_features']:
        X = transform_features(X)
    y = data[[TARGET_LABEL]].astype(int).reset_index(drop=True)

    data_plot.plot_feature_distribution(data, "redshift")

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

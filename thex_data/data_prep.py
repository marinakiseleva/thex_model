"""
data_prep
Prepares data before running through a classifier. Contains functionality to pull down data, filter data, enhances features, and split into training and testing sets.
"""

from sklearn.model_selection import train_test_split

from .data_init import collect_data
from .data_filter import *
from .data_transform import transform_features
from .data_consts import TARGET_LABEL, ORIG_TARGET_LABEL


def get_data(col_list, **data_filters):
    """
    Pull in data and filter based on different biases and corrections: group transient types, fitler to passed-in columns, keep only rows with at least 1 valid value, filter to most frequent classes, sub-sample each class to same number, and take difference between wavelengths to make new features
    :param data_columns: List of columns to filter data on
    :param **data_filters: Mapping of data filters to values passed in by user
    """

    df = collect_data()
    # Relabel label column
    df[TARGET_LABEL] = df[ORIG_TARGET_LABEL]

    # Remove rows with NULL labels
    df = df[~df[TARGET_LABEL].isnull()]

    # Drop empty class labels
    df = df[df[TARGET_LABEL] != ""]

    df = filter_columns(df.copy(), col_list, data_filters['incl_redshift'])

    # Drop row with any NULL values (after columns have been filtered)
    df.dropna(axis=0, inplace=True)

    df = drop_conflicts(df)

    df = filter_class_labels(df, data_filters['class_labels'])

    # Keep classes with minimum number of samples (Not too few)
    df = filter_class_size(df,
                           data_filters['min_class_size'],
                           data_filters['class_labels'])

    # Randomly subsample any over-represented classes down to passed-in value
    # (Not too many)
    df = sub_sample(df,
                    data_filters['subsample'],
                    data_filters['class_labels'])

    if df.shape[0] == 0:
        raise ValueError(
            "\nNo data to run model on. Try changing data filters or limiting number of features. Note: Running on all columns will not work since no data spans all features.\n")

    return df.reset_index(drop=True)


def get_source_target_data(data_columns, **data_filters):
    """
    Gets data split into source and target; but not yet split into training and testing
    """
    data = get_data(col_list=data_columns, **data_filters)
    X = data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
    if data_filters['transform_features']:
        X = transform_features(X)

    y = data[[TARGET_LABEL]]

    return X, y

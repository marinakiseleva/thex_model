"""
data_prep
Prepares data before running through a classifier. Contains functionality to pull down data, filter data, enhances features, and split into training and testing sets.
"""

from .data_init import collect_data
from .data_filter import *
from .data_transform import transform_features
from .data_consts import TARGET_LABEL, ORIG_TARGET_LABEL


def get_data(col_list, data_filters):
    """
    Pull in data and filter based on different biases and corrections: group transient types, fitler to passed-in columns, keep only rows with at least 1 valid value, filter to most frequent classes, sub-sample each class to same number, and take difference between wavelengths to make new features
    :param data_columns: List of columns to filter data on
    :param **data_filters: Mapping of data filters to values passed in by user
    """
    if data_filters['case_code'] is not None:
        col_list.append("case_code")

    df = data_filters['data']
    if data_filters['data'] is None:
        df = collect_data(data_filters['data_file'])

    # Relabel label column
    df[TARGET_LABEL] = df[ORIG_TARGET_LABEL]

    # Remove rows with NULL labels
    df = df[~df[TARGET_LABEL].isnull()]

    # Drop empty class labels
    df = df[df[TARGET_LABEL] != ""]

    df = filter_columns(df.copy(deep=True), col_list)

    if data_filters['nb']:
        # Naive bayes: if it has at leat 1 non-null item, do not drop
        df.dropna(thresh=1, subset=col_list, inplace=True)
    else:
       # Drop row with any NULL values (after columns have been filtered)
        df.dropna(axis=0, inplace=True)

    df = drop_conflicts(df)

    if data_filters['case_code'] is not None:
        df= df.loc[df['case_code'].isin(data_filters['case_code'])] 
        df.drop(labels='case_code',axis=1,inplace=True)


    if df.shape[0] == 0:
        raise ValueError("\nNo data to run model on.\n")

    return df.reset_index(drop=True)


def get_source_target_data(data_columns, data_filters):
    """
    Gets data split into source and target; but not yet split into training and testing
    """
    data = get_data(data_columns, data_filters)
    X = data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
    if data_filters['transform_features']:
        X = transform_features(X)

    y = data[[TARGET_LABEL]]

    return X, y

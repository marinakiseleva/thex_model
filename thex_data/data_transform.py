"""
Enhance features by scaling and changing them
"""
import pandas as pd
import numpy as np


def transform_features(df):
    """
    Computes feature transformations. Enhances existing features by computing new representations of same features.
    :param df: DataFrame of features
    """
    df = derive_diffs(df)
    # df = derive_reciprocals(df)
    df = scale_data(df)
    return df


def scale_data(df):
    """
    Scale each feature between 0 and 1 for consistent PCA
    """
    for index, colname in enumerate(list(df)):
        max_col = df[colname].max()
        min_col = df[colname].min()
        df[colname] = df[colname].apply(
            lambda x: (x - min_col) / (max_col - min_col))
    return df


def derive_diffs(df):
    """
    Subtracts adjacent color bands from one another and saves result in new columns, labeled as differences between two
    :param df: DataFrame of features
    """
    features = list(df)
    print(features)
    for index, colname1 in enumerate(features):
        if index < len(features) - 1:
            colname2 = df.columns[index + 1]  # Get next column
            val1 = df[colname1]
            val2 = df[colname2]
            new_col_name = colname2 + "_minus_" + colname1
            df[new_col_name] = val2 - val1

    return df


def derive_reciprocals(df):
    """
    Compute reciprocals of each column, 1/x
    """
    for index, colname in enumerate(list(df)):
        df[colname + "_reciprocal"] = df[colname] / 1
    return df


def derive_logs(df):
    """
    Compute log of each column
    """
    for index, colname in enumerate(list(df)):
        df[colname + "_log"] = np.log(df[colname])
    return df

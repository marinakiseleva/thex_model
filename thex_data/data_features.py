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
    df = derive_reciprocals(df)
    return df


def derive_diffs(df):
    """
    Subtracts adjacent color bands from one another and saves result in new columns, labeled as differences between two
    :param df: DataFrame of features
    """
    features = list(df)
    for index, colname1 in enumerate(features):
        if index < len(features) - 1:
            colname2 = df.columns[index + 1]  # Get next column
            # Only do colors for spectral data (mag = magnitude)
            val1 = df[colname1]
            val2 = df[colname2]
            # Only take difference if adjacent columns are both magnitudes
            if val1 is not None and val2 is not None and (('mag' in colname2 and 'mag' in colname1) or ('KCORRECT' in colname2 and 'KCORRECT' in colname1)):
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

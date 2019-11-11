"""
data_transform
Enhance features by scaling and transforming them
"""
from thex_data.data_consts import adjacent_mags


def transform_features(df):
    """
    Computes feature transformations. Enhances existing features by computing new representations of same features.
    :param df: DataFrame of features
    """
    df = derive_diffs(df)
    # df = scale_data(df)
    return df


def scale_data(df):
    """
    Scale each feature between 0 and 1 for consistent PCA
    :param df: DataFrame of features
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
    for index, colname1 in enumerate(features):
        if colname1 in adjacent_mags:
            colname2 = adjacent_mags[colname1]
            if colname2 in features:
                primary_mag = df[colname1]
                prev_mag = df[colname2]
                new_col_name = colname2 + "_minus_" + colname1
                print("Adding new column " + new_col_name)
                df[new_col_name] = primary_mag - prev_mag

    return df

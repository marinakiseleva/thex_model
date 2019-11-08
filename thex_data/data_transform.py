"""
data_transform
Enhance features by scaling and transforming them
"""
from thex_data.data_consts import mag_cols


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
        if index < len(features) - 1:
            colname2 = df.columns[index + 1]  # Get next column
            if (colname1 in mag_cols or 'mag' in colname1) and (colname2 in mag_cols or 'mag' in colname2):
                val1 = df[colname1]
                val2 = df[colname2]
                new_col_name = colname2 + "_minus_" + colname1
                print("Adding new column " + new_col_name)
                df[new_col_name] = val2 - val1

    return df

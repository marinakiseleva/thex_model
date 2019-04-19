"""
Enhance features by scaling and changing them
"""
import pandas as pd
import numpy as np
from thex_data.data_consts import mag_cols, TARGET_LABEL
from thex_data.data_clean import convert_str_to_list


def transform_features(df):
    """
    Computes feature transformations. Enhances existing features by computing new representations of same features.
    :param df: DataFrame of features
    """
    df = derive_diffs(df)
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
    for index, colname1 in enumerate(features):
        if index < len(features) - 1:
            colname2 = df.columns[index + 1]  # Get next column
            if colname1 in mag_cols and colname2 in mag_cols:
                val1 = df[colname1]
                val2 = df[colname2]
                new_col_name = colname2 + "_minus_" + colname1
                df[new_col_name] = val2 - val1

    return df


def convert_class_vectors(df, class_labels):
    """
    Convert labels of TARGET_LABEL column in passed-in DataFrame to class vectors
    """
    # Convert labels to class vectors, with 1 meaning it has that class, and 0
    # does not
    rows_list = []
    for df_index, row in df.iterrows():
        class_vector = [0] * len(class_labels)
        cur_classes = convert_str_to_list(row[TARGET_LABEL])
        for class_index, c in enumerate(class_labels):
            if c in cur_classes:
                class_vector[class_index] = 1
        rows_list.append([class_vector])
    class_vectors = pd.DataFrame(rows_list, columns=[TARGET_LABEL])
    return class_vectors

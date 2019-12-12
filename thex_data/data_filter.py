"""
data_filter
Functionality for filtering data
 - sub_sample: sub sampling over-represented classes
 - one_all: Keep only certain classes unique and convert rest to 'Other', in order to attempt a pseudo one-vs-all classification
 - filter_columns: keeps only certain columns; feature selection
 - filter_top_classes: keep only X most popular classes to reduce # of classes and ensure all classes have enough data
"""

import pandas as pd

from .data_consts import cat_code, TARGET_LABEL
from .data_print import *
import utilities.utilities as util


def filter_class_labels(df, class_labels):
    """
    Keep rows that have label in self.class_labels
    :param df: DataFrame of features and TARGET_LABEL
    """
    # Keep rows that have a label in self.class_labels.
    if class_labels is None:
        return df
    keep_indices = []
    for df_index, row in df.iterrows():
        list_classes = util.convert_str_to_list(row[TARGET_LABEL])
        for c in list_classes:
            if c in class_labels:
                keep_indices.append(df_index)
                break

    return df.loc[keep_indices, :].reset_index(drop=True)


def drop_conflicts(df):
    """
    Drop rows where classes belong to classes in different disjoint groups. This requires special handling not yet implemented. 
    """
    conflict_labels = ["_CONFUSION", "_CONFLICT",
                       "__CONFLICT_CASES", "_UNCLEAR_LABELS", "_IGNORED_LABELS"]
    keep_indices = []
    for df_index, row in df.iterrows():
        list_classes = util.convert_str_to_list(row[TARGET_LABEL])
        keep = True
        for c in list_classes:
            if c in conflict_labels:
                keep = False
        if keep:
            keep_indices.append(df_index)

    return df.loc[keep_indices, :]


def sub_sample(df, count, classes):
    """
    Sub-samples over-represented class
    :param df: DataFrame to subsample
    :param count: Randomly subsample all classes to this #; if class has less than or equal to this #, then just leave it
    :param classes: classes to filter on.
    """
    if count is None:
        return df

    if classes is None:
        # Infer unique classes from data
        unique_class_groups = list(df[TARGET_LABEL].unique())
        unique_classes = set()
        for group in unique_class_groups:
            for class_name in group:
                unique_classes.add(class_name)
        unique_class_list = list(unique_classes)
    else:
        unique_class_list = classes

    # Filter each class in list
    subsampled_df = pd.DataFrame()
    for class_label in unique_class_list:
        keep_indices = []
        for index, row in df.iterrows():
            class_list = util.convert_str_to_list(row[TARGET_LABEL])
            if class_label in class_list:
                keep_indices.append(index)
        df_class = df.loc[keep_indices, :].reset_index(drop=True)
        num_rows = df_class.shape[0]  # number of rows
        if num_rows > count:
            # Reduce to the count number
            df_class = df_class.sample(n=count)
        subsampled_df = pd.concat([subsampled_df, df_class])

    return subsampled_df


def filter_columns(df, col_list, incl_redshift):
    """
    Filters columns down to those passed in as col_list (+ target label and redshift if selected)
    """
    # Filter DataFrame down to these columns
    col_list = col_list + [TARGET_LABEL]
    if incl_redshift:
        col_list.append('redshift')
    df = df[col_list]
    return df


def filter_class_size(df, X):
    """
    Keep classes with at least X points
    :param df: DataFrame of features and TARGET_LABEL
    :param X: Number of classes to keep
    :return: DataFrame of original format, with only top X classes of data
    """
    if X is None:
        return df
    filtered_df = pd.DataFrame()
    unique_classes = list(df[TARGET_LABEL].unique())
    for class_label in unique_classes:
        df_class = df[df[TARGET_LABEL] == class_label]
        num_rows = df_class.shape[0]  # number of rows
        if num_rows > X:
            # Keep since it has enough rows
            filtered_df = pd.concat([filtered_df, df_class])

    return filtered_df


def filter_data(df):
    """
    Filters DataFrame to keep only rows that have at least 1 valid value in the features
    :param df: DataFrame of features and TARGET_LABEL
    """
    df = df.reset_index(drop=True)
    cols = list(df)
    if 'redshift' in cols:
        cols.remove('redshift')
    cols.remove(TARGET_LABEL)
    # Only check for NULLs on real feature columns
    df_filtered = pd.DataFrame(df[cols].reset_index(drop=True))
    # Indices of data that do not have all NULL or 0 values
    non_null_indices = df_filtered.loc[
        (~df_filtered.isnull().all(1)) & (~(df_filtered == 0).all(1))].index

    return df.iloc[non_null_indices]

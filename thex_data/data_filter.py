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


def sub_sample(df, count):
    """
    Sub-samples over-represented class
    :param df: the dataframe to manipulate
    :param count: number to set all classes to; if class has less than this, then just leave it
    """
    if count is None:
        return df
    subsampled_df = pd.DataFrame()
    unique_classes = list(df[TARGET_LABEL].unique())
    for class_code in unique_classes:
        df_class = df[df[TARGET_LABEL] == class_code]
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
    if col_list is None:
        raise ValueError('Need to pass in values for col_list')

    # Filter DataFrame down to these columns
    col_list = col_list + [TARGET_LABEL]
    if incl_redshift:
        col_list.append('redshift')
    df = df[col_list]
    return df


def one_all(df, keep_classes):
    """
    Convert all classes not in keep_classes to 'Other' 
    :param df: DataFrame of features and TARGET_LABEL
    :param keep_classes: list of class NAMES to keep unique
    """
    if keep_classes is None:
        return df
    class_codes = [cat_code[name] for name in keep_classes]
    one = df[df[TARGET_LABEL].isin(class_codes)]  # keep unique classes
    # Set remaining classes to other
    rem = df.loc[~df[TARGET_LABEL].isin(class_codes)].copy()
    rem[TARGET_LABEL] = cat_code['Other']
    df = pd.concat([one, rem])
    return df


def get_class_counts(df):
    """
    Returns Frame of class count
    :param df: DataFrame of features and TARGET_LABEL
    """
    class_counts = df.groupby(TARGET_LABEL).count()
    class_counts['avg_count'] = class_counts.mean(axis=1)
    class_counts = class_counts.sort_values(
        'avg_count', ascending=False).reset_index()
    return class_counts


def filter_top_classes(df, X):
    """
    Keep X most frequent classes
    :param df: DataFrame of features and TARGET_LABEL
    :param X: Number of classes to keep
    :return: DataFrame of original format, with only top X classes of data
    """
    if X is None:
        return df
    top_classes = get_class_counts(df).head(X)
    top_classes = list(top_classes[TARGET_LABEL])
    return df.loc[df[TARGET_LABEL].isin(top_classes)]


def filter_class_size(df, X):
    """
    Keep classes with at least X points
    :param df: DataFrame of features and TARGET_LABEL
    :param X: Number of classes to keep
    :return: DataFrame of original format, with only top X classes of data
    """
    class_counts = get_class_counts(df)
    min_classes = class_counts.loc[class_counts['avg_count'] >= X]
    min_classes = list(min_classes[TARGET_LABEL])
    return df.loc[df[TARGET_LABEL].isin(min_classes)]


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

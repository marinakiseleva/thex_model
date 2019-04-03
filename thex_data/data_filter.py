"""
The functions below involve filtering data down to normalize it
 - sub_sample: sub sampling over-represented classes
 - one_all: Keep only certain classes unique and convert rest to 'Other', in order to attempt a pseudo one-vs-all classification
 - filter_columns: keeps only certain columns; feature selection
 - filter_top_classes: keep only X most popular classes to reduce # of classes and ensure all classes have enough data
"""

import pandas as pd
from .data_consts import cat_code, TARGET_LABEL
from .data_print import *


def sub_sample(df, count, col_val):
    """
    Sub-samples over-represented class
    :param df: the dataframe to manipulate
    :param count: number to set all classes to; if class has less than this, then just leave it
    """
    subsampled_df = pd.DataFrame()
    t_values = list(df[col_val].unique())
    # iterate through each claimed type group in dataset
    for ctg in t_values:
        cur_ctg = df[df[col_val] == ctg]  # rows with this claimed type group
        num_rows = cur_ctg.shape[0]  # number of rows
        if num_rows > count:
            # Reduce to the count number
            cur_ctg = cur_ctg.sample(n=count)
        subsampled_df = pd.concat([subsampled_df, cur_ctg])

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

    # print_features_used(col_list)
    df = df[col_list]
    # Convert transient class to number code
    df[TARGET_LABEL] = df[TARGET_LABEL].apply(lambda x: int(cat_code[x]))
    return df


def one_all(df, keep_classes):
    """
    Convert all classes not in keep_classes to 'Other' 
    :param keep_classes: list of class NAMES to keep unique
    """
    if keep_classes is None:
        return df
    class_codes = [cat_code[name] for name in keep_classes]
    one = df[df[TARGET_LABEL].isin(class_codes)]  # keep unique classes
    rem = df.loc[~df[TARGET_LABEL].isin(class_codes)].copy()  # set to other
    rem[TARGET_LABEL] = 100
    df = pd.concat([one, rem])
    return df


def get_popular_classes(df, top=5):
    """
    Gets most popular classes by frequency. Returns list of 'top' most popular class class codes
    :param top: # of classes to return
    """
    ttype_counts = df.groupby(TARGET_LABEL).count()
    ttype_counts['avg_count'] = ttype_counts.mean(axis=1)
    ttype_counts = ttype_counts.sort_values(
        'avg_count', ascending=False).reset_index().head(top)

    return list(ttype_counts[TARGET_LABEL])


def filter_top_classes(df, top=5):
    top_classes = get_popular_classes(df, top=top)
    return df.loc[df[TARGET_LABEL].isin(top_classes)]


def filter_data(df):
    """
    Filters DataFrame to keep only rows that have at least 1 valid value in the features
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

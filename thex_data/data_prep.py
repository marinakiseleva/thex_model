"""
data_prep:
 manages all functions that filters down and prepares the data for machine learning:
 - sub sampling
 - one-vs-all classification
 - feature selection
 - splitting source vs target


"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_consts import cat_code, TARGET_LABEL, DATA_PATH
from .data_print import *
from .data_init import collect_data
from .data_clean import *


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


def split_train_test(X, y):
    # Split Training and Testing Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=2)
    return X_train, X_test, y_train, y_test


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
    if 'AllWISE_IsVar' in col_list:
        col_list.remove('AllWISE_IsVar')

    print_features_used(col_list)
    df = df[col_list]
    # Convert transient class to number code
    df[TARGET_LABEL] = df[TARGET_LABEL].apply(lambda x: int(cat_code[x]))
    return df


def one_all(df, keep_classes):
    """
    Convert all classes not in keep_classes to 'Other' 
    :param keep_classes: list of class NAMES to keep unique
    """
    class_codes = [cat_code[name] for name in keep_classes]
    one = df[df[TARGET_LABEL].isin(class_codes)]  # keep unique classes
    rem = df.loc[~df[TARGET_LABEL].isin(class_codes)].copy()  # set to other
    rem[TARGET_LABEL] = 100
    df = pd.concat([one, rem])
    return df


def derive_diffs(df):
    """
    Creates colors of spectral data by subtracting different bandwidth spectra from one another - subtracts adjacent color bands from one another
    """
    features = list(df)
    diff_df = pd.DataFrame()
    for index, colname1 in enumerate(features):
        if index < len(features) - 1:
            colname2 = df.columns[index + 1]  # Get next column
            # Only do colors for spectral data (mag = magnitude)
            val1 = df[colname1]
            val2 = df[colname2]
            if val1 is not None and val2 is not None and (('mag' in colname2 and 'mag' in colname1) or ('KCORRECT' in colname2 and 'KCORRECT' in colname1)):
                new_col_name = colname2 + "_minus_" + colname1
                diff_df[new_col_name] = val2 - val1
            elif 'mag' not in colname1 and "KCORRECT" not in colname1:
                diff_df[colname1] = df[colname1]
            if index == len(features) - 2:
                diff_df[colname2] = df[colname2]

    return diff_df


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


def get_data(col_list, incl_redshift):
    """
    Pull in data and filter based on different biases and corrections: group transient types, fitler to passed-in columns, keep only rows with at least 1 valid value, filter to most frequent classes, sub-sample each class to same number, and take difference between wavelengths to make new features 
    """
    # Go back one directory, because we are in thex_model
    df = collect_data()
    df = group_cts(df)

    df = filter_columns(df.copy(), col_list, incl_redshift)
    # df = filter_data(df)
    df.dropna(axis=0, inplace=True)

    df = filter_top_classes(df, top=10)
    # df = one_all(df, ['TDE', 'Ia', 'II P'])

    # Randomly subsample any over-represented classes down to 100
    df = sub_sample(df, count=500, col_val=TARGET_LABEL)

    # Derive colors from data, and keep only colors
    # df = derive_diffs(df.copy())

    if df.shape[0] == 0:
        print("No data to run model on. Try changing filters or limiting number of features. Note: Running on all columns will not work since no data spans all features.")
        sys.exit()

    print_class_counts(df)
    return df


def get_source_target_data(data_columns, incl_redshift):
    data = get_data(col_list=data_columns, incl_redshift=incl_redshift)
    X = data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
    y = data[[TARGET_LABEL]].astype(int).reset_index(drop=True)
    return X, y


def get_train_test(data_columns, incl_redshift, split=0.4):
    """
    cat: Boolean on whether to record classes as category strings (True) or category codes (False)
    """
    X, y = get_source_target_data(data_columns, incl_redshift)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)

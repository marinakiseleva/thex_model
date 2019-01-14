"""
data_prep:
 manages all functions that filters down and prepares the data for machine learning:
 - sub sampling
 - one-vs-all classification
 - feature selection
 - splitting source vs target


"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_maps import cat_code
from .data_init import collect_data
from .data_clean import group_cts


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


def set_up_target(df_rs_band):
    """
    Sets up Target dataframe. Converts claimedtype_group strings to numeric codes
    using cat_code map above
    """
    df_analyze = df_rs_band.copy()
    # Split out claimedtype_group into new dataframe
    y = df_analyze.claimedtype_group.to_frame(name='group')
    # Use category code numbers from cat_code dict, instead of strings
    y['cat_code'] = y.apply(lambda row:  cat_code[row.group], axis=1)
    y = y.drop('group', axis=1)
    return y


def set_up_source(df_analyze):
    X = df_analyze.copy()
    X = X.drop(['redshift', 'claimedtype_group'], axis=1)

    return X


def split_train_test(X, y):
    # Split Training and Testing Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=2)
    return X_train, X_test, y_train, y_test


def filter_columns(df, col_list, incl_redshift=False):
    """
    Filters columns down to those passed in as col_list PLUS redshift and claimedtype_group
    If col_list is None, a set of columns is created
    """
    print("Filtering on columns " + str(col_list))
    if col_list is None:
        full_list = ['NSA_ELPETRO_KCORRECT', 'Kmag']
        col_list = [col for col in list(df) if any(
            col_val in col and "_e_" not in col for col_val in full_list)]

    # Filter DataFrame down to these columns
    sel_cols = col_list + ['claimedtype_group']
    if incl_redshift:
        sel_cols.append('redshift')
    df = df[sel_cols]

    # Convert transient class to number code
    df['transient_type'] = df.apply(lambda row:  int(
        cat_code[row.claimedtype_group]), axis=1)
    df.drop(['claimedtype_group'], axis=1, inplace=True)

    return df


def one_all(df, keep_cols, col):
    one = df[df['claimedtype_group'].isin(keep_cols)]  # keep unique classes
    rem = df.loc[~df['claimedtype_group'].isin(keep_cols)].copy()  # set to other
    rem[col] = 'Other'
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
    Gets most popular classes, top is the # of classes
    """
    ttype_counts = df.groupby('transient_type').count()
    ttype_counts['avg_count'] = ttype_counts.mean(axis=1)
    ttype_counts = ttype_counts.sort_values(
        'avg_count', ascending=False).reset_index().head(top)

    return list(ttype_counts['transient_type'])


def filter_top_classes(df, top=5):
    top_classes = get_popular_classes(df, top=top)
    return df.loc[df['transient_type'].isin(top_classes)]


def get_data(col_list, incl_redshift=False, file='THEx-catalog.v0_0_3.fits'):
    cur_path = os.path.dirname(__file__)
    # Go back one directory, because we are in thex_model
    df = collect_data(cur_path + "/../../../data/" + file)
    df = group_cts(df)
    df = filter_columns(df, col_list, incl_redshift)

    df = filter_top_classes(df, top=5)

    # Randomly subsample any over-represented classes down to 100
    df = sub_sample(df, count=400, col_val='transient_type')

    # df.dropna(inplace=True)

    # Derive colors from data, and keep only colors
    # df = derive_diffs(df.copy())

    return df


def get_train_test(col_list, incl_redshift=False, file='THEx-catalog.v0_0_3.fits'):
    """
    Initialize data for Naive Bayes classifier using 70/30 split for train/test
    """
    df = get_data(col_list, incl_redshift, file)

    # Split into train and test
    train = df.sample(frac=0.7, random_state=200)
    test = df.drop(train.index)

    print("Training set size: " + str(train.shape[0]))
    print("Testing set size: " + str(test.shape[0]))
    return train, test

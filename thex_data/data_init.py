"""
Handles accessing and updating data.
Initializes data from initial pull. Filters down list of columns based on user-input. 
"""


import pandas as pd
import numpy as np
from astropy.table import Table

from thex_data.data_consts import EXCLUDE_COLS, ORIG_TARGET_LABEL


def collect_cols(cols, col_matches, data_file):
    """
    Return all columns that contain at least one string in the list of 
    col_matches or all columns in cols list
    :param cols: List of columns to filter on
    :param col_matches: List of strings by which columns will be selected. 
    For example: AllWISE will use all AlLWISE columns.
    """
    col_list = []
    if cols is not None:
        if not isinstance(cols, list):
            raise TypeError("cols in collect_cols must be a list.")
        col_list = cols
    else:
        # Make list of column/feature names; exlcude error columns
        all_cols = list(collect_data(data_file))
        if col_matches is not None:
            if not isinstance(col_matches, list):
                raise TypeError("col_matches in collect_cols must be a list.")
            for column in all_cols:
                if any(match_value in column for match_value in col_matches):
                    col_list.append(column)
        else:
            col_list = all_cols
            if 'redshift' in col_list:
                col_list.remove('redshift')
            if ORIG_TARGET_LABEL in col_list:
                col_list.remove(ORIG_TARGET_LABEL)

    # Drop all non-numeric columns
    # column_list_numeric = set()
    # for c in col_list:
    #     if not any(col in c for col in EXCLUDE_COLS):
    #         column_list_numeric.add(c)  # Keep only numeric columns

    return list(col_list)


def collect_data(data_file):
    """ 
    Sets up Data Object using data 
    :return: Pandas DataFrame created from data_file data 
    """
    dat = Table.read(data_file, format='fits')
    df_bytes = dat.to_pandas()  # Convert to pandas dataframe
    df = pd.DataFrame()     # Init empty dataframe for converted types

    # Convert byte columns to strings
    for column in df_bytes:
        if df_bytes[column].dtype == np.dtype('object'):
            df[column + "_str"] = df_bytes[column].str.decode("utf-8")
            df[column] = df[column + "_str"].copy(deep=True)
            df.drop(column + "_str", axis=1, inplace=True)
        else:
            df[column] = df_bytes[column]
    # Drop infinity values.
    df = df[~df.isin([np.inf, -np.inf]).any(1)]
    return df

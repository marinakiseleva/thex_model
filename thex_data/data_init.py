import pandas as pd
from astropy.table import Table
import numpy as np

from thex_data.data_consts import DATA_PATH, drop_cols


"""
Initializes data from initial pull. Filters down list of columns based on user-input.
"""


def collect_cols(cols, col_matches):
    """
    Return all columns that contain at least one string in the list of col_matches or all columns in cols list
    :param cols: List of columns to filter on
    :param col_matches: List of strings by which columns will be selected. For example: AllWISE will use all AlLWISE columns.
    """
    col_list = []
    if cols is not None:
        if not isinstance(cols, list):
            raise TypeError("cols in collect_cols must be a list.")
        col_list = cols
    else:
        # Make list of column/feature names; exlcude error columns
        all_cols = list(collect_data())
        if col_matches is not None:
            if not isinstance(col_matches, list):
                raise TypeError("col_matches in collect_cols must be a list.")
            for column in all_cols:
                if any(match_value in column for match_value in col_matches):
                    col_list.append(column)
        else:
            col_list = all_cols
            col_list.remove('redshift')

    # Drop all non-numeric columns
    column_list_numeric = []
    for c in col_list:
        if not any(col in c for col in drop_cols):
            column_list_numeric.append(c)  # Keep only numeric columns
    if 'redshift' in column_list_numeric:
        raise ValueError(
            "Do not include redshift in list of columns -- instead at it in through the flag.")
    return column_list_numeric


def collect_data():
    """ 
    Sets up Data Object using data 
    """
    # if '.npy' in DATA_PATH:
    #     # Read in npy file
    #     # np.load(DATA_PATH)
    #     data = np.load(DATA_PATH).item()
    #     print(data)
    dat = Table.read(DATA_PATH, format='fits')
    df_bytes = dat.to_pandas()  # Convert to pandas dataframe
    df = pd.DataFrame()     # Init empty dataframe for converted types

    # Convert byte columns to strings
    for column in df_bytes:
        if df_bytes[column].dtype == np.dtype('object'):
            df[column + "_str"] = df_bytes[column].str.decode("utf-8")
            df[column] = df[column + "_str"].copy()
            df.drop(column + "_str", axis=1, inplace=True)
        else:
            df[column] = df_bytes[column]

    return df

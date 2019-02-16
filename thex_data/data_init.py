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
    :param cols: Names of columns to filter on
    :param col_matches: String by which columns will be selected. For example: AllWISE will use all AlLWISE columns.
    """
    col_list = []
    if cols is not None:
        col_list = cols
    else:
        # Make list of column/feature names; exlcude error columns
        all_cols = list(collect_data())
        column_list = []
        if col_matches is not None:
            for column in all_cols:
                if any(match_value in column for match_value in col_matches):
                    column_list.append(column)
        else:
            # Use all columns in data set with numeric values
            column_list = all_cols

        # Drop all non-numeric columns
        column_list_numeric = []
        for c in column_list:
            if not any(col in c for col in drop_cols):
                column_list_numeric.append(c)  # Keep only numeric columns

    return column_list_numeric


def collect_data():
    """ 
    Sets up Data Object using data 
    """
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

    # Filter on only confirmed claimed types
    df = df.loc[df.is_confirmed_host == 1]
    return df

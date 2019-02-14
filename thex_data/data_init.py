import pandas as pd
from astropy.table import Table
import numpy as np

from thex_data.data_consts import DATA_PATH, drop_cols


def collect_cols(col_matches):
    """
    Return all columns that contain at least one string in the list of col_matches
    """
    all_cols = list(collect_data())
    # Make list of column/feature names; exlcude error columns
    column_list = []
    for column in all_cols:
        if any(match_value in column for match_value in col_matches):
            column_list.append(column)

    # Drop all non-numeric columns
    column_list_numeric = []
    for c in column_list:
        if not any(col in c for col in drop_cols):
            column_list_numeric.append(c)

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

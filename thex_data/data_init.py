import pandas as pd
from astropy.table import Table
import numpy as np

from thex_data.data_consts import DATA_PATH


def collect_cols(col_names):
    df = collect_data()
    # Make list of column/feature names; exlcude _e_ (errors)
    col_list = [col for col in list(df) if any(
                col_val in col and "_e_" not in col for col_val in col_names)]
    return col_list


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

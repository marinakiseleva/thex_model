"""
data_clean:
 manages all functions that cleans and prepares the data before filtering

"""

from .data_consts import groupings, TARGET_LABEL
import numpy as np


def group_cts(df):
    """
    Normalizes claimed type (transient type) into a specific category (one of the values in the groupings map). If claimed type is not in map, it is removed. Only considers 1-1 mappings, does not use galaxies that have more than 1
    :param df: Pandas DataFrame of galaxy/transient data. Must have column 'claimedtype' with transient type
    :return df: Returns Pandas DataFrame with new column TARGET_LABEL, which has normalized transient type for each galaxy. 
    """
    new_column = TARGET_LABEL

    # Group claimed types into supergroups, defined in groupings dict
    df[new_column] = df['claimedtype'].apply(
        lambda x: groupings[x] if x in groupings else None)

    # Dataframe of claimed types that do not have group
    # ungrouped_types = list(set(t_df.claimedtype.unique()) - set(groupings.keys()))

    # Drop rows with no grouped claimed type
    df = df[~df[new_column].isnull()]

    return df


def fill_nulls(df):
    """
    Fills NULL values in dataframe by assigning average value in the column 
    """
    for col_name in list(df):
        df[col_name].fillna((df[col_name].mean()), inplace=True)

        mask = df[col_name] != np.inf
        df.loc[~mask, col_name] = df.loc[mask, col_name].mean()

    return df

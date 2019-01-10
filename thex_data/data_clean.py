"""
data_clean:
 manages all functions that cleans and prepares the data before filtering

"""

from .data_maps import groupings


def group_cts(t_df):
    """
    Normalizes claimed type (transient type) into a specific category (one of the values in the groupings map). If claimed type is not in map, it is removed.
    :param t_df: Pandas DataFrame of galaxy/transient data. Must have column 'claimedtype' with transient type
    :return t_df: Returns Pandas DataFrame with new column claimedtype_group, which has normalized transient type for each galaxy. 
    """
    new_column = 'claimedtype_group'
    # Group claimed types into supergroups, defined in groupings dict
    # Only considers 1-1 mappings, does not use galaxies that have more than 1
    # transient class possible
    t_df[new_column] = t_df.apply(lambda row:
                                  groupings[row.claimedtype]
                                  if row.claimedtype in groupings
                                  else None,
                                  axis=1)

    # Dataframe of claimed types that do not have group
    # ungrouped_types = list(set(t_df.claimedtype.unique()) - set(groupings.keys()))

    # Drop rows with no grouped claimed type
    t_df = t_df[~t_df[new_column].isnull()]

    return t_df


def fill_nulls(t_df):
    """
    Fills NULL values in dataframe by assigning average value in the column 
    """
    for col_name in list(t_df):
        t_df[col_name].fillna((t_df[col_name].mean()), inplace=True)
    return t_df

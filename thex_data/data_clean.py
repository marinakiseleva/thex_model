"""
data_clean:
 manages all functions that cleans and prepares the data before filtering

"""

from .data_consts import groupings, TARGET_LABEL
from .data_consts import class_to_subclass as hierarchy
import numpy as np
from hmc import hmc


def init_tree():
    hmc_hierarchy = hmc.ClassHierarchy("TTypes")
    for parent in hierarchy.keys():
        # hierarchy maps parents to children, so get all children
        list_children = hierarchy[parent]
        for child in list_children:
            # Nodes are added with child parent pairs
            try:
                hmc_hierarchy.add_node(child, parent)
            except ValueError as e:
                print(e)
    return hmc_hierarchy


def assign_levels(tree, mapping, node, level):
    mapping[str(node)] = level
    for child in tree._get_children(node):
        assign_levels(tree, mapping, child, level + 1)
    return mapping


def convert_str_to_list(input_string):
    """
    Convert string to list
    """
    l = input_string.split(",")
    return [item.strip(' ') for item in l]


def group_by_tree(df):
    """
    Normalized claimed type (transient type) - assign lowest level in tree.
    """
    tree = init_tree()
    node_depths = {}
    ltree = assign_levels(tree, node_depths, tree.root, 1)

    for index, row in df.iterrows():
        orig_labels = convert_str_to_list(row['claimedtype'])
        max_depth = 0
        min_label = None
        for label in orig_labels:
            if label in node_depths and node_depths[label] > max_depth:
                max_depth = node_depths[label]
                min_label = label

        df.at[index, 'claimedtype'] = min_label
    return df


def group_cts(df):
    """
    Normalizes claimed type (transient type) into a specific category (one of the values in the groupings map). If claimed type is not in map, it is removed. Only considers 1-1 mappings, does not use galaxies that have more than 1
    :param df: Pandas DataFrame of galaxy/transient data. Must have column 'claimedtype' with transient type
    :return df: Returns Pandas DataFrame with new column TARGET_LABEL, which has normalized transient type for each galaxy. 
    """

    # Group claimed types into supergroups, defined in groupings dict
    df[TARGET_LABEL] = df['claimedtype'].apply(
        lambda x: groupings[x] if x in groupings else None)

    # Dataframe of claimed types that do not have group
    # ungrouped_types = list(set(t_df.claimedtype.unique()) - set(groupings.keys()))

    # Drop rows with no grouped claimed type
    df = df[~df[TARGET_LABEL].isnull()]

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

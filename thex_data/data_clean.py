"""
data_clean:
 manages all functions that cleans and prepares the data before filtering

"""

from .data_consts import groupings, TARGET_LABEL, cat_code
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


def group_by_tree(df, transform_labels):
    """
    Normalized claimed type (transient type) - assign lowest level in tree. If claimed type is not in map, it is removed. Only considers 1-1 mappings, does not use galaxies that have more than 1. Defines new TARGET_LABEL column. 
    :param df: DataFrame of values and labels. Must have column 'claimedtype' with transient type.
    :return df: Returns Pandas DataFrame with new column TARGET_LABEL, which has normalized transient type for each 
    """
    if transform_labels == False:
        df[TARGET_LABEL] = df['claimedtype']
        df = df[~df[TARGET_LABEL].isnull()]
        return df

    tree = init_tree()
    node_depths = assign_levels(tree, {}, tree.root, 1)

    for index, row in df.iterrows():
        orig_labels = convert_str_to_list(row['claimedtype'])
        max_depth = 0
        min_label = None
        for label in orig_labels:
            if label in groupings:
                # Groups labels into mapped definitions, ie. 1c -> Ic
                label = groupings[label]

            if label in node_depths and node_depths[label] > max_depth:
                max_depth = node_depths[label]
                min_label = label

        df.at[index, 'claimedtype'] = min_label

    df[TARGET_LABEL] = df['claimedtype'].apply(
        lambda x: groupings[x] if x in groupings else None)
    df = df[~df[TARGET_LABEL].isnull()]

    # Convert transient class to number code
    df[TARGET_LABEL] = df[TARGET_LABEL].apply(lambda x: int(cat_code[x]))

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

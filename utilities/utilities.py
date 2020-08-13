import os
import matplotlib.pyplot as plt
from hmc import hmc

from thex_data.data_consts import ROOT_DIR, TREE_ROOT, TARGET_LABEL, UNDEF_CLASS


# Hierarchy Utilities

def init_tree(hierarchy):
    print("\n\nConstructing Class Hierarchy Tree...")
    hmc_hierarchy = hmc.ClassHierarchy(TREE_ROOT)
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
    """
    Assigns level to each node based on level in hierarchical tree. The lower it is in the tree, the larger the level. The level at the root is 1.
    :return: Dict from class name to level number.
    """
    mapping[str(node)] = level
    for child in tree._get_children(node):
        assign_levels(tree, mapping, child, level + 1)
    return mapping


def add_unspecified_labels_to_data(y, class_levels):
    """
    Add unspecified label for each tree parent in data's list of labels
    :param class_levels: self.class_levels from model
    """
    for index, row in y.iterrows():
        # Iterate through all class labels for this label
        max_depth = 0  # Set max depth to determine what level is undefined
        for label in convert_str_to_list(row[TARGET_LABEL]):
            if label in class_levels:
                max_depth = max(class_levels[label], max_depth)
        # Max depth will be 0 for classes unhandled in hierarchy.
        if max_depth > 0:
            # Add Undefined label for any nodes at max depth
            for label in convert_str_to_list(row[TARGET_LABEL]):
                if label in class_levels and class_levels[label] == max_depth:
                    add = ", " + UNDEF_CLASS + label
                    y.iloc[index] = y.iloc[index] + add
    return y


# Plotting Utilities

def annotate_plot(ax, x, y, annotations):
    """
    Adds count to top of each bar in bar plot
    """
    index = 0
    for xy in zip(x, y):
        # Get class count of current index
        count = str(annotations[index])
        ax.annotate(count, xy=xy, textcoords='data', ha='center',
                    va='bottom', fontsize=8, rotation=-90)
        index += 1


def save_plot(model_dir, file_name, bbox_inches=None, fig=None):
    """
    Saves plot (by model name and passed-in title)
    :param model_dir: Directory name to save to
    :param file_name: Save plot with this name
    :[optional] param bbox_inches: Optional parameter for savefig
    :[optional] param fig: will do fig.savefig instead of plt.savefig
    """
    plt.tight_layout()

    if fig is not None:
        fig.savefig(model_dir + "/" + clean_str(file_name) +
                    ".pdf", bbox_inches=bbox_inches)
    else:
        plt.savefig(model_dir + "/" + clean_str(file_name) +
                    ".pdf", bbox_inches=bbox_inches)


def display_and_save_plot(model_dir, file_name, bbox_inches=None, fig=None):
    """
    Saves plot by passed-in file name, and displays.
    :param model_dir: Directory name to save to
    :param file_name: Save plot with this name
    :[optional] param bbox_inches: Optional parameter for savefig
    :[optional] param fig: will do fig.savefig instead of plt.savefig
    """
    save_plot(model_dir, file_name, bbox_inches, fig)
    plt.show()


# File utilities
def init_file_directories(name):
    """
    Initialize new directory by incrementing value of last directory.
    """
    # Create output directory
    name = clean_str(name)
    output_dir = ROOT_DIR + "/output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dirs = os.listdir(output_dir)

    max_num = 0
    for directory in dirs:
        if name in directory:
            num = directory.split(name)[1]
            # Numeric value of that output dir
            num = int(num) if num != '' and num.isdigit() else 0
            max_num = max(max_num, num)

    # Create new dir with next number
    new_dir = output_dir + "/" + name + str(max_num + 1)
    os.mkdir(new_dir)
    return new_dir


def convert_str_to_list(input_string):
    """
    Convert string to list
    """
    l = input_string.split(",")
    return [item.strip(' ') for item in l]


def clean_str(text):
    """
    Remove unnecessary characters from text (in order to save it as valid file name)
    """
    replace_strs = ["\n", " ", ":", ".", ",", "/"]
    for r in replace_strs:
        text = text.replace(r, "_")
    text = text.replace('%', "perc")
    return text

import os
import shutil
from textwrap import wrap
import matplotlib.pyplot as plt
from thex_data.data_consts import ROOT_DIR


# Plotting Utilities

def annotate_plot(ax, x, y, annotations):
    """
    Adds count to top of each bar in bar plot
    """
    class_index = 0
    for xy in zip(x, y):
        # Get class count of current class_index
        count = str(annotations[class_index])
        ax.annotate(count, xy=xy, textcoords='data', ha='center',
                    va='bottom', fontsize=8, rotation=-90)
        class_index += 1


def save_plot(model_dir, title, ax, bbox_inches=None, fig=None):
    """
    Saves plot (by model name and passed-in title)
    :param model_dir: Directory name to save to
    :param title: String title of plot, used to save 
    :param ax: Axis
    :[optional] param bbox_inches: Optional parameter for savefig
    :[optional] param fig: will do fig.savefig instead of plt.savefig
    """
    if ax is None:
        plt.title('\n'.join(wrap(title, 60)))
    else:
        ax.set_title('\n'.join(wrap(title, 60)))
    title = clean_str(title)
    plt.tight_layout()

    if fig is not None:
        fig.savefig(model_dir + "/" + title, bbox_inches=bbox_inches)
        fig.savefig('samplefigure', bbox_extra_artists=None,
                    bbox_inches='tight')

    else:
        plt.savefig(model_dir + "/" + title, bbox_inches=bbox_inches)


def display_and_save_plot(model_dir, title, ax, bbox_inches=None, fig=None):
    """
    Saves plot (by model name and passed-in title) and displays.
    :param model_dir: Directory name to save to
    :param title: String title of plot, used to save 
    :param ax: Axis
    :[optional] param bbox_inches: Optional parameter for savefig
    :[optional] param fig: will do fig.savefig instead of plt.savefig
    """
    save_plot(model_dir, title, ax, bbox_inches, fig)
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
    return text

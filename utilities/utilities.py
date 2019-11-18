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


def save_plot(model_name, title, ax, bbox_inches=None, fig=None, extra_artists=None):
    """
    Saves plot (by model name and passed-in title)
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

    file_dir = ROOT_DIR + "/output/" + clean_str(model_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    if fig is not None:
        fig.savefig(file_dir + "/" + title, bbox_inches=bbox_inches)
        fig.savefig('samplefigure', bbox_extra_artists=extra_artists,
                    bbox_inches='tight')

    else:
        plt.savefig(file_dir + "/" + title, bbox_inches=bbox_inches)


def display_and_save_plot(model_name, title, ax, bbox_inches=None, fig=None):
    """
    Saves plot (by model name and passed-in title) and displays.
    :param title: String title of plot, used to save 
    :param ax: Axis
    :[optional] param bbox_inches: Optional parameter for savefig
    :[optional] param fig: will do fig.savefig instead of plt.savefig
    """
    save_plot(model_name, title, ax, bbox_inches, fig)
    plt.show()


# File utilities
def init_file_directories(name):
    # Create output directory
    if not os.path.exists(ROOT_DIR + "/output"):
        os.mkdir(ROOT_DIR + "/output")

    file_dir = ROOT_DIR + "/output/" + clean_str(name)
    # Clear old output directories, if they exist
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.mkdir(file_dir)


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

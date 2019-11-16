import os
import shutil
from thex_data.data_consts import ROOT_DIR


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

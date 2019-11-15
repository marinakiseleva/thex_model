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

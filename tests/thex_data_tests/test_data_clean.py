import sys
import unittest
sys.path.append("../..")

from thex_data.data_clean import *
from thex_data.data_consts import TARGET_LABEL, class_to_subclass, ORIG_TARGET_LABEL

"""

Tests data_init

Run tests with:
python -m unittest

"""


class TestDataInit(unittest.TestCase):

    def test_group_by_tree(self):
        test_data = pd.DataFrame([
            [20, 15, "Ia"], [5, 8, "CC"], [6, 8, "CC"], [21, 16, "Ia"], [18, 16, "Ia"]], columns=['feature1', 'feature2', ORIG_TARGET_LABEL])
        output_data = group_by_tree(test_data, False)
        print(output_data)

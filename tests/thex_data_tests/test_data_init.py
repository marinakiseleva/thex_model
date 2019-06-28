import sys
import unittest
sys.path.append("../..")

from thex_data.data_consts import drop_cols
from thex_data.data_init import *

"""

Tests data_init

Run tests with:
python -m unittest

"""


class TestDataInit(unittest.TestCase):

    def test_collect_cols(self):
        filtered_columns = collect_cols(cols=None, col_matches=None)
        for existing_col in filtered_columns:
            for bad_col in drop_cols:
                if bad_col == existing_col:
                    self.fail("Did not filter out column in drop_cols.")

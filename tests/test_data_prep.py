import sys
import unittest
import pandas as pd
sys.path.append("..")

from thex_bayes import data_prep


class TestDataPrepFunctions(unittest.TestCase):

    def test_derive_diffs(self):
        df = pd.DataFrame([[2, 4, 6], [2, 4, 6]], columns=['mag1', 'mag2', 'mag3'])
        real_df = pd.DataFrame([[2, 2], [2, 2]], columns=[
                               'mag2_minus_mag1', 'mag3_minus_mag2'])
        func_df = data_prep.derive_diffs(df)
        self.assertTrue(real_df.equals(func_df))


if __name__ == '__main__':
    unittest.main()

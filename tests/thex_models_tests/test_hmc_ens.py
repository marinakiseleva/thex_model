import sys
import unittest
import pandas as pd
sys.path.append("..")
import numpy as np

from models.clus_hmc_ens_model.clus_hmc_ens_model import *
from thex_data.data_consts import TARGET_LABEL

"""
Run tests with:
python -m unittest 

"""


class TestHMCENSLogic(unittest.TestCase):

    def setUp(self):
        self.test_hmc_ens = CLUSHMCENS()  # Test instance
        # DataFrame that represents labels (y), each row is class vector
        self.test_df = pd.DataFrame(
            [[[1, 0, 1]], [[0, 0, 1]], [[0, 0, 1]]], columns=[TARGET_LABEL])
        self.full_test_df = pd.DataFrame(
            [[4, 6, [1, 0, 1]], [1, 3, [0, 0, 1]], [8, 9, [0, 0, 1]]], columns=['feature1', 'feature2', TARGET_LABEL])

    # def test_get_variance(self):
    #     self.test_hmc_ens.class_weights = np.array([3, 0, 1])
    #     v = self.test_hmc_ens.get_variance(self.full_test_df, 'feature1', 4.5)
    #     print(v)
    #     print(float(v))
    #     self.assertEqual(v, 0)

        # var greater than is 0, but variance less than is:

    def test_split_samples(self):
        # self, labeled_samples, feature, value, label_set=None
        samples_greater, samples_less = self.test_hmc_ens.split_samples(
            self.full_test_df, 'feature1', 4.5)

        actual_greater = pd.DataFrame(
            [[8, 9, [0, 0, 1]]], columns=['feature1', 'feature2', TARGET_LABEL]).reset_index(drop=True)
        actual_less = pd.DataFrame(
            [[4, 6, [1, 0, 1]], [1, 3, [0, 0, 1]]], columns=['feature1', 'feature2', TARGET_LABEL]).reset_index(drop=True)

        self.assertTrue(actual_greater.equals(samples_greater.reset_index(drop=True)))
        self.assertTrue(actual_less.equals(samples_less.reset_index(drop=True)))

    def test_is_unambiguous(self):
        estimated = self.test_hmc_ens.is_unambiguous(self.test_df)
        self.assertEqual(False, estimated)
        test_df2 = pd.DataFrame(
            [[[0, 0, 1]], [[0, 0, 1]]], columns=[TARGET_LABEL])
        estimated = self.test_hmc_ens.is_unambiguous(test_df2)
        self.assertEqual(True, estimated)

    def test_get_mean_vector(self):
        estimated = self.test_hmc_ens.get_mean_vector(self.test_df)
        expected = np.array([0.33333333, 0.,      1.])
        # allclose checks if 2 arrays are equal within tolerance (because they are
        # floats other comparisons don't work)
        self.assertTrue(np.allclose(estimated, expected))

    def test_majority_class(self):
        self.test_hmc_ens.class_weights = [1, 1, 1]
        maj_class = self.test_hmc_ens.majority_class(self.full_test_df)
        expected = [.33, 0, 1]
        for index, v1 in enumerate(maj_class):
            self.assertEqual(round(maj_class[index], 2), expected[index])

    def test_get_weighted_distance(self):
        # Test logic alone
        a = np.array([0, .65, 8])
        b = np.array([1, 6, 7])
        self.test_hmc_ens.class_weights = np.array([0, 3, 1])
        estimate = self.test_hmc_ens.get_weighted_distance(a, b)
        self.assertEqual(round(estimate, 2), 9.32)


if __name__ == '__main__':
    unittest.main()

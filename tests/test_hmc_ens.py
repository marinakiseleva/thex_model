import sys
import unittest
import pandas as pd
sys.path.append("..")
import numpy as np

from models.clus_hmc_ens_model.clus_hmc_ens_model import *


class TestHMCENSLogic(unittest.TestCase):

    def test_get_weighted_distance(self):
        # Test logic alone
        a = np.array([0, .65, 8])
        b = np.array([1, 6, 7])
        w = np.array([0, 3, 1])
        estimate = (np.dot(w, (np.square(a - b))))**(1 / 2)
        self.assertEqual(round(estimate, 2), 9.32)


if __name__ == '__main__':
    unittest.main()

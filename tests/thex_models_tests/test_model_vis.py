import sys
import unittest
import numpy as np
import pandas as pd
sys.path.append("../../")

from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS
from mainmodel.helper_compute import *
from mainmodel.helper_plotting import *

from models.binary_model.binary_model import BinaryModel
from models.ind_model.ind_model import OvAModel
from models.multi_model.multi_model import MultiModel


import warnings


"""

Tests Visualization and Performance Measures of models

Test functions in MainModel through BinaryModel, MultiModel, and OvAModel, since MainModel is an abstract class

Run tests with:
python -m unittest

"""


class TestModelBalancedPurity(unittest.TestCase):

    def generate_data(self, original, num_datapoints):
        data = []
        for i in range(num_datapoints):
            noise = np.random.normal(loc=0, scale=0.1, size=3)
            data.append(original + noise)
        return np.array(data)

    def setUp(self):
        warnings.filterwarnings('ignore')

        num_type1 = 20
        num_type2 = 10
        c1 = self.generate_data([0.2, 0.3, .9], num_type1)
        c2 = self.generate_data([0.3, 0.1, .8], num_type2)
        data_X = np.concatenate((c1, c2))
        fake_X = pd.DataFrame(data_X, columns=['f1', 'f2', 'f3'])
        fake_Y = pd.DataFrame(["Ia"] * num_type1 + ["CC"] *
                              num_type2, columns=[TARGET_LABEL])

        fake_data = [fake_X, fake_Y]

        self.class_labels = ["TDE", "II", "Ib/c"]
        # Rows for max prob range
        preds = [[0.9, 0.1, 0, "TDE"],
                 [0.95, 0.05, 0, "TDE"],
                 [0.92, 0, 0.08, "TDE"],
                 [0.1, 0.9, 0, "TDE"],
                 [0.1, 0.9, 0, "II"],
                 [0, 0.9, 0.1, "II"],
                 [0.91, 0.09, 0.1, "Ib/c"],
                 [0, 0.9, 0.1, "Ib/c"],
                 [0, 0.05, 0.95, "Ib/c"],
                 [0, 0.01, 0.99, "Ib/c"],
                 # .8 to .9
                 [0.81, 0.09, 0.01, "TDE"],
                 # Preds for lower prob ranges
                 [0.65, 0.35, 0, "TDE"],
                 [0.6, 0.4, 0, "TDE"],
                 [0.35, 0.65, 0, "TDE"],
                 [0.35,  0, 0.65, "TDE"],
                 [0.35, 0, 0.65, "TDE"],
                 [0.1, 0.6, 0.3, "II"],
                 [0.1, 0.3, 0.6, "II"],
                 [0.1, 0.3, 0.6, "II"],
                 [0.6, 0.4, 0, "Ib/c"],
                 [0.4, 0.6, 0, "Ib/c"],
                 [0, 0.35, 0.65, "Ib/c"],
                 ]

        self.agg_results = preds
        self.test_model = BinaryModel(data=fake_data,
                                      class_labels=self.class_labels)

    def test_bp_binary_get_class_stats(self):
        TDE_stats = bp_binary_get_class_stats(target_class="TDE",
                                              prob_range=[0.9, 1],
                                              rows=self.agg_results,
                                              class_labels=self.class_labels)
        self.assertEqual(TDE_stats, [3, 1, 4, 6])  # TP, FP, P, N
        II_stats = bp_binary_get_class_stats(target_class="II",
                                             prob_range=[0.9, 1],
                                             rows=self.agg_results,
                                             class_labels=self.class_labels)

        self.assertEqual(II_stats, [2, 2, 2, 8])  # TP, FP, P, N

    def test_bp_binary_get_assignments(self):
        """
        Gets [TP, FP, total P, total N] for each class, in order of class_labels.
        """
        assgs = bp_binary_get_assignments(preds=self.agg_results,
                                          class_labels=self.class_labels,
                                          bin_size=0.1)
        self.assertEqual(assgs[8], [[1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        self.assertEqual(assgs[9], [[3, 1, 4, 6], [2, 2, 2, 8], [2, 0, 4, 6]])

    def test_bp_binary_aggregate_assignments(self):
        assgs = bp_binary_get_assignments(preds=self.agg_results,
                                          class_labels=self.class_labels,
                                          bin_size=0.1)
        aggassgs = bp_binary_aggregate_assignments(assignments=assgs,
                                                   class_labels=self.class_labels,
                                                   num_bins=10)
        a = np.array_equal(aggassgs[9], np.array(
            [[3, 1, 4, 6], [2, 2, 2, 8], [2, 0, 4, 6]]))
        self.assertTrue(a)
        b = np.array_equal(aggassgs[8], np.array(
            [[4, 1, 5, 6], [2, 2, 2, 9], [2, 0, 4, 7]]))
        self.assertTrue(b)

    def test_bp_binary_get_range_bps(self):
        """
        bp_binary_get_range_bps gets balanced purity for each probability range threshold, and each class. For class A, get balanced purity at probs >=10%, probs>=20% and so on. Return 10 dicts in list, in order of prob ranges, each dict is  target/predicted class name to balanced purity at threshold.
        """
        assgs = bp_binary_get_assignments(preds=self.agg_results,
                                          class_labels=self.class_labels,
                                          bin_size=0.1)
        aggassgs = bp_binary_aggregate_assignments(assignments=assgs,
                                                   class_labels=self.class_labels,
                                                   num_bins=10)
        range_bps = bp_binary_get_range_bps(aggassgs, self.class_labels)

        bp9 = np.around(list(range_bps[9].values()), decimals=2)
        a = np.array_equal(bp9, [0.82, 0.80, 1.0])
        self.assertTrue(a)
        bp8 = np.around(list(range_bps[8].values()), decimals=2)
        b = np.array_equal(bp8, [0.83, 0.82, 1.0])
        self.assertTrue(b)

        bp6 = np.around(list(range_bps[6].values()), decimals=2)

        self.assertTrue(np.array_equal(bp6, [0.78, 0.72, 0.62]))

    def test_get_binary_balanced_purity_ranges(self):
        """
        get_binary_balanced_purity_ranges
        Get balanced purity for each class, for each range of probabilities, for binary classifier.  Returns dict. of class names to balanced purity at each prob threshold (10 thresholds)
        """
        bps = get_binary_balanced_purity_ranges(preds=self.agg_results,
                                                class_labels=self.class_labels,
                                                bin_size=0.1)
        TDE_bps = np.around(list(bps["TDE"]), decimals=2)
        self.assertTrue(np.array_equal(TDE_bps[6:], [0.78, 0.83, 0.83, 0.82]))

    def test_bp_get_class_stats(self):
        """
        bp_get_class_stats
        Return list in order of class_labels, where each value corresponds to number of samples of that class predicted as target class with a probability in the range prob_range. Last value of the list is total # of rows with true label of target_class
        """

        TDE_stats = bp_get_class_stats(target_class="TDE",
                                       prob_range=[0.9, 1],
                                       rows=self.agg_results,
                                       class_labels=self.class_labels)
        self.assertEqual(TDE_stats, [3, 0, 1, 4])
        II_stats = bp_get_class_stats(target_class="II",
                                      prob_range=[0.9, 1],
                                      rows=self.agg_results,
                                      class_labels=self.class_labels)

        self.assertEqual(II_stats, [1, 2, 1, 2])

    def test_bp_get_assignments(self):
        """
        Gets # of samples of each class assigned to each target class, by probability range.
        Multiclass.
        """

        assgs = bp_get_assignments(preds=self.agg_results,
                                   class_labels=self.class_labels,
                                   bin_size=0.1)
        self.assertEqual(assgs[8], [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(assgs[9], [[3, 0, 1, 4], [1, 2, 1, 2], [0, 0, 2, 4]])

    def test_bp_aggregate_assignments(self):
        assgs = bp_get_assignments(preds=self.agg_results,
                                   class_labels=self.class_labels,
                                   bin_size=0.1)
        aggassgs = bp_aggregate_assignments(assignments=assgs,
                                            class_labels=self.class_labels,
                                            num_bins=10)

        self.assertTrue(np.array_equal(
            aggassgs[6], [[6, 0, 2, 10], [2, 3, 2, 5], [2, 2, 3, 7]]))
        self.assertTrue(np.array_equal(
            aggassgs[8], [[4, 0, 1, 5], [1, 2, 1, 2], [0, 0, 2, 4]]))
        self.assertTrue(np.array_equal(
            aggassgs[9], [[3, 0, 1, 4], [1, 2, 1, 2], [0, 0, 2, 4]]))

    def test_bp_get_range_bps(self):
        """
        list, in order of ranges; each list contains dict from target/predicted class name to balanced purity at threshold.
        """
        assgs = bp_get_assignments(preds=self.agg_results,
                                   class_labels=self.class_labels,
                                   bin_size=0.1)
        aggassgs = bp_aggregate_assignments(assignments=assgs,
                                            class_labels=self.class_labels,
                                            num_bins=10)
        range_bps = bp_get_range_bps(aggassgs, self.class_labels)

        bp9 = np.around(list(range_bps[9].values()), decimals=2)
        self.assertTrue(np.array_equal(bp9, [0.75, 0.67, 1.0]))

        bp6 = np.around(list(range_bps[6].values()), decimals=2)
        self.assertTrue(np.array_equal(bp6, [0.68, 0.55, 0.42]))

        # get_balanced_purity_ranges(preds, class_labels, bin_size)

        # compute_binary_balanced_purity

        # compute_balanced_purity

        # get_puritys_and_comps

        # get_completeness_ranges

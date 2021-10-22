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

Tests to make sure calculations of experimental performance (purity, completeness, balanced purity, aggregated balanced purity, etc.) is all correct.

Test functions in MainModel through BinaryModel, MultiModel, and OvAModel, since MainModel is an abstract class

Run tests with:
python -m unittest

"""


class TestModelMetrics(unittest.TestCase):

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

        self.class_counts = self.test_model.get_class_counts(
            pd.DataFrame(preds).rename({3: "transient_type"}, axis=1))

    ##############################################################
    ########## BP Ranges                #########################
    ##############################################################

    def test_bp_binary_get_class_stats(self):
        TDE_stats = bp_binary_get_class_stats(target_class="TDE",
                                              prob_range=[0.9, 1],
                                              rows=self.agg_results,
                                              class_labels=self.class_labels)
        self.assertEqual(TDE_stats, [3, 1])  # TP, FP, P, N
        II_stats = bp_binary_get_class_stats(target_class="II",
                                             prob_range=[0.9, 1],
                                             rows=self.agg_results,
                                             class_labels=self.class_labels)

        self.assertEqual(II_stats, [2, 2])  # TP, FP, P, N

    def test_bp_binary_get_assignments(self):
        """
        Gets [TP, FP, total P, total N] for each class, in order of class_labels.
        """
        assgs = bp_binary_get_assignments(preds=self.agg_results,
                                          class_labels=self.class_labels,
                                          bin_size=0.1)
        self.assertEqual(assgs[8], [[1, 0], [0, 0], [0, 0]])
        self.assertEqual(assgs[9], [[3, 1], [2, 2], [2, 0]])

    def test_bp_binary_aggregate_assignments(self):
        assgs = bp_binary_get_assignments(preds=self.agg_results,
                                          class_labels=self.class_labels,
                                          bin_size=0.1)
        aggassgs = bp_binary_aggregate_assignments(assignments=assgs,
                                                   class_labels=self.class_labels,
                                                   num_bins=10)
        a = np.array_equal(aggassgs[9], np.array(
            [[3, 1], [2, 2], [2, 0]]))
        self.assertTrue(a)
        b = np.array_equal(aggassgs[8], np.array(
            [[4, 1], [2, 2], [2, 0]]))
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
        range_bps = bp_binary_get_range_bps(aggassgs,
                                            self.class_labels,
                                            total_class_counts=self.class_counts)
        bp9 = np.around(list(range_bps[9].values()), decimals=2)
        a = np.array_equal(bp9, [0.78, 0.77, 1.0])
        self.assertTrue(a)

        bp6 = np.around(list(range_bps[6].values()), decimals=2)

        bp6_TDE = round((6 / 10) / ((6 / 10) + (2 / 12)), 2)
        bp6_II = round((3 / 5) / ((3 / 5) + (4 / 17)), 2)
        bp6_Ibc = round((3 / 7) / ((3 / 7) + (4 / 15)), 2)
        self.assertTrue(np.array_equal(bp6, [bp6_TDE, bp6_II, bp6_Ibc]))

    def test_get_binary_balanced_purity_ranges(self):
        """
        get_binary_balanced_purity_ranges
        Get balanced purity for each class, for each range of probabilities, for binary classifier.  Returns dict. of class names to balanced purity at each prob threshold (10 thresholds)
        """
        bps = get_binary_balanced_purity_ranges(preds=self.agg_results,
                                                class_labels=self.class_labels,
                                                bin_size=0.1,
                                                total_class_counts=self.class_counts)
        TDE_bps = np.around(list(bps["TDE"]), decimals=2)
        bp6_TDE = round((6 / 10) / ((6 / 10) + (2 / 12)), 2)
        bp7_TDE = round((4 / 10) / ((4 / 10) + (1 / 12)), 2)
        bp8_TDE = round((4 / 10) / ((4 / 10) + (1 / 12)), 2)
        bp9_TDE = round((3 / 10) / ((3 / 10) + (1 / 12)), 2)
        self.assertTrue(np.array_equal(
            TDE_bps[6:], [bp6_TDE, bp7_TDE, bp8_TDE, bp9_TDE]))

    def test_bp_get_class_stats(self):
        """
        bp_get_class_stats
        # of rows with true label of target_class
        Return list in order of class_labels, where each value corresponds to number of samples of that class predicted as target class with a probability in the range prob_range. Last value of the list is total
        """

        TDE_stats = bp_get_class_stats(target_class="TDE",
                                       prob_range=[0.9, 1],
                                       rows=self.agg_results,
                                       class_labels=self.class_labels)
        self.assertEqual(TDE_stats, [3, 0, 1])
        II_stats = bp_get_class_stats(target_class="II",
                                      prob_range=[0.9, 1],
                                      rows=self.agg_results,
                                      class_labels=self.class_labels)

        self.assertEqual(II_stats, [1, 2, 1])

    def test_bp_get_assignments(self):
        """
        Gets # of samples of each class assigned to each target class, by probability range.
        Multiclass.
        """

        assgs = bp_get_assignments(preds=self.agg_results,
                                   class_labels=self.class_labels,
                                   bin_size=0.1)
        self.assertEqual(assgs[8], [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(assgs[9], [[3, 0, 1], [1, 2, 1], [0, 0, 2]])

    def test_bp_aggregate_assignments(self):
        assgs = bp_get_assignments(preds=self.agg_results,
                                   class_labels=self.class_labels,
                                   bin_size=0.1)
        aggassgs = bp_aggregate_assignments(assignments=assgs,
                                            class_labels=self.class_labels,
                                            num_bins=10)

        self.assertTrue(np.array_equal(
            aggassgs[6], [[6, 0, 2], [2, 3, 2], [2, 2, 3]]))
        self.assertTrue(np.array_equal(
            aggassgs[8], [[4, 0, 1], [1, 2, 1], [0, 0, 2]]))
        self.assertTrue(np.array_equal(
            aggassgs[9], [[3, 0, 1], [1, 2, 1], [0, 0, 2]]))

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
        range_bps = bp_get_range_bps(aggassgs, self.class_labels, self.class_counts)

        bp9 = np.around(list(range_bps[9].values()), decimals=2)
        # 0.75 = (0.75) / (0.75+ (1/4))
        # [0.75, 0.67, 1.0]
        self.assertTrue(np.array_equal(bp9, [0.68, 0.62, 1.0]))

        bp6 = np.around(list(range_bps[6].values()), decimals=2)
        self.assertTrue(np.array_equal(bp6, [0.68, 0.55, 0.42]))

    def test_get_balanced_purity_ranges(self):

        bps = get_balanced_purity_ranges(preds=self.agg_results,
                                         class_labels=self.class_labels,
                                         bin_size=0.1,
                                         total_class_counts=self.class_counts)

        TDE_bps = np.around(list(bps["TDE"]), decimals=2)
        self.assertTrue(np.array_equal(TDE_bps[6:], [0.68, 0.74, 0.74, 0.68]))

        II_bps = np.around(list(bps["II"]), decimals=2)
        self.assertTrue(np.array_equal(II_bps[6:], [0.55, 0.62, 0.62, 0.62]))

        Ibc_bps = np.around(list(bps["Ib/c"]), decimals=2)
        self.assertTrue(np.array_equal(Ibc_bps[6:], [0.42, 1.0, 1.0, 1.0]))

    ##############################################################
    ########## Empirical Probs           #########################
    ##############################################################
    def test_multi_empirical_probabilities(self):
        """
        Get empirical probability for each prob range; same as balanced purity by prob range.
        """

        eps = get_multi_emp_prob_rates(preds=self.agg_results,
                                       class_labels=self.class_labels,
                                       bin_size=0.2,
                                       total_class_counts=self.class_counts)

        TDE_eps = np.around(list(eps["TDE"]), decimals=2)
        self.assertEqual(TDE_eps[0], 0.06)
        self.assertEqual(TDE_eps[1], 1)
        self.assertEqual(TDE_eps[4], 0.74)

        II_eps = np.around(list(eps["II"]), decimals=2)
        self.assertEqual(II_eps[0], 0)
        self.assertEqual(II_eps[1], 0.62)
        self.assertEqual(II_eps[4], 0.62)

        Ibc_eps = np.around(list(eps["Ib/c"]), decimals=2)
        self.assertEqual(Ibc_eps[0], 0.32)
        self.assertEqual(Ibc_eps[1], 0)
        self.assertEqual(Ibc_eps[4], 1)

    def test_binary_empirical_probabilities(self):
        """
        Get empirical probability for each prob range; same as balanced purity by prob range.
        """

        eps = get_binary_emp_prob_rates(preds=self.agg_results,
                                        class_labels=self.class_labels,
                                        bin_size=0.2,
                                        total_class_counts=self.class_counts)

        TDE_eps = np.around(list(eps["TDE"]), decimals=2)
        self.assertEqual(TDE_eps[3], round((2 / 10) / ((2 / 10) + (1 / 12)), 2))
        self.assertEqual(TDE_eps[4], round((4 / 10) / ((4 / 10) + (1 / 12)), 2))

        II_eps = np.around(list(eps["II"]), decimals=2)
        self.assertEqual(II_eps[3], round((1 / 5) / ((1 / 5) + (2 / 17)), 2))
        self.assertEqual(II_eps[4], round((2 / 5) / ((2 / 5) + (2 / 17)), 2))

        Ibc_eps = np.around(list(eps["Ib/c"]), decimals=2)
        self.assertEqual(Ibc_eps[3], round((1 / 7) / ((1 / 7) + (4 / 15)), 2))
        self.assertEqual(Ibc_eps[4], round((2 / 7) / ((2 / 7)), 2))

    ##############################################################
    ########## BP Averages                #########################
    ##############################################################
    def test_compute_balanced_purity(self):
        """
        Get balanced purity for each class, return as 2 dicts.
        balanced purity = TPR/ (TPR+FPR)
        """
        # Binary
        purities = compute_binary_balanced_purity(preds=self.agg_results,
                                                  class_labels=self.class_labels)

        self.assertEqual(round(purities["TDE"], 2), 0.78)
        self.assertEqual(round(purities["II"], 2), 0.72)
        self.assertEqual(round(purities["Ib/c"], 2), 0.62)
        # Multi
        purities = compute_balanced_purity(preds=self.agg_results,
                                           class_labels=self.class_labels,
                                           model_name="multi")

        self.assertEqual(round(purities["TDE"], 2), 0.68)
        self.assertEqual(round(purities["II"], 2), 0.55)
        self.assertEqual(round(purities["Ib/c"], 2), 0.42)

    def test_get_puritys_and_comps(self):
        """
        get_puritys_and_comps returns dict from class name to purity and a dict from class name to completeness.
        """
        class_metrics = {"TDE": {"TP": 10, "FP": 20, "FN": 5, "TN": 40}}
        p, c = get_puritys_and_comps(class_metrics)
        self.assertEqual(round(p["TDE"], 2), 0.33)
        self.assertEqual(round(c["TDE"], 2), 0.67)

    def test_get_completeness_ranges(self):
        # true_positives, totals = range_metrics[class_name]
        range_metrics = {}
        TPS = [20, 10]
        Totals = [50, 20]
        range_metrics["TDE"] = [TPS, Totals]
        class_counts = {}
        class_counts["TDE"] = 70

        cr = get_completeness_ranges(class_counts, range_metrics, "TDE")
        print(cr)
        self.assertEqual(round(cr[0], 2), round(30 / 70, 2))
        self.assertEqual(round(cr[1], 2), round(10 / 70, 2))

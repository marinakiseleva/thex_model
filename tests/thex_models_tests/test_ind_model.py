import sys
import unittest
import numpy as np
import pandas as pd
sys.path.append("../../")

from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS

from models.ind_model.ind_model import IndModel


import warnings


"""

Tests IndModel

Also test functions in MainModel through IndModel, since MainModel is an abstract class

Run tests with:
python -m unittest

"""


class TestIndModel(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore')

        def generate_data(original, num_datapoints):
            data = []
            for i in range(num_datapoints):
                noise = np.random.normal(loc=0, scale=0.1, size=3)
                data.append(original + noise)

            return np.array(data)
        num_type1 = 20
        num_type2 = 10
        c1 = generate_data([0.2, 0.3, .9], num_type1)
        c2 = generate_data([0.3, 0.1, .8], num_type2)
        data_X = np.concatenate((c1, c2))
        fake_X = pd.DataFrame(data_X, columns=['f1', 'f2', 'f3'])
        fake_Y = pd.DataFrame(["Ia"] * num_type1 + ["CC"] *
                              num_type2, columns=[TARGET_LABEL])

        fake_data = [fake_X, fake_Y]

        self.orig_X = fake_X
        self.orig_y = fake_Y
        self.test_model = IndModel(data=fake_data)

    ##########################################################################
    # MainModel functions
    ##########################################################################
    def test_filter_labels(self):
        orig_set = ["Ia", "CC", "Ia-91bg", UNDEF_CLASS + "Ia", ]
        new_set = self.test_model.filter_labels(orig_set)
        expected = ["CC", "Ia-91bg", UNDEF_CLASS + "Ia"]
        self.assertEqual(sorted(expected), sorted(new_set))

        orig_set = ["Ia", "CC"]
        new_set = self.test_model.filter_labels(orig_set)
        expected = ["Ia", "CC"]
        self.assertEqual(sorted(expected), sorted(new_set))

    def test_filter_data(self):
        X, y = self.test_model.filter_data(X=self.test_model.X,
                                           y=self.test_model.y,
                                           min_class_size=6,
                                           max_class_size=2,
                                           class_labels=["Unspecified Ia", "Unspecified CC"])
        self.assertEqual(X.shape[0], 4)

        X, y = self.test_model.filter_data(X=self.test_model.X,
                                           y=self.test_model.y,
                                           min_class_size=21,
                                           max_class_size=40,
                                           class_labels=["Unspecified Ia", "Unspecified CC"])
        self.assertEqual(X.shape[0], 0)

    def test_add_unspecified_labels_to_data(self):

        new_y = self.test_model.add_unspecified_labels_to_data(self.orig_y)
        unspec_ia_count = 0
        for index, row in new_y.iterrows():
            if "Unspecified Ia" in row[TARGET_LABEL]:
                unspec_ia_count += 1
        self.assertEqual(unspec_ia_count, 20)

    def test_get_class_labels(self):

        labels = self.test_model.get_class_labels(user_defined_labels=None,
                                                  y=self.test_model.y,
                                                  N=4)
        expected = sorted(['Unspecified CC', 'CC', 'Ia', 'Unspecified Ia'])
        self.assertEqual(sorted(labels), expected)

        # With labels defined
        labels = self.test_model.get_class_labels(user_defined_labels=["CC", "Unspecified CC"],
                                                  y=self.test_model.y,
                                                  N=4)
        expected = sorted(['Unspecified CC', 'CC'])
        self.assertEqual(sorted(labels), expected)

        # With higher min class requirement
        labels = self.test_model.get_class_labels(user_defined_labels=None,
                                                  y=self.test_model.y,
                                                  N=12)
        expected = sorted(['Ia', 'Unspecified Ia'])
        self.assertEqual(sorted(labels), expected)

        # Both min class req and labels
        labels = self.test_model.get_class_labels(user_defined_labels=["CC", "Unspecified CC"],
                                                  y=self.test_model.y,
                                                  N=12)
        self.assertEqual(sorted(labels), [])

    ##########################################################################
    # IndModel functions
    ##########################################################################

    def test_relabel_class_data(self):
        relabeled_y = self.test_model.relabel_class_data("Ia", self.test_model.y)
        expected = True
        for index, row in relabeled_y.iterrows():
            if index >= 0 and index <= 19:
                # Should be 1, because it was Ia
                if row[TARGET_LABEL] != 1:
                    expected = False
                    break
        self.assertEqual(expected, True)

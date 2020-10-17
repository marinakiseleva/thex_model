import sys
import unittest
import numpy as np
import pandas as pd
sys.path.append("../../")

from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS

from models.ind_model.ind_model import OvAModel


import warnings


"""

Tests OvAModel

Also test functions in MainModel through OvAModel, since MainModel is an abstract class

Run tests with:
python -m unittest

"""


class TestOvAModel(unittest.TestCase):

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

        self.orig_X = fake_X
        self.orig_y = fake_Y
        self.test_model = OvAModel(data=fake_data)

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

    def test_get_class_counts(self):
        counts = self.test_model.get_class_counts(self.test_model.y)
        self.assertEqual(counts["Unspecified Ia"], 20)
        self.assertEqual(counts["Unspecified CC"], 10)

    def test_scale_data(self):
        np.set_printoptions(precision=2, suppress=True)
        c1 = self.generate_data([30, 100, 20], 4)
        c2 = self.generate_data([20, 60, 20], 2)
        X_train = np.concatenate((c1, c2))
        X_train = pd.DataFrame(X_train, columns=['f1', 'f2', 'f3'])

        c1 = self.generate_data([30, 100, 20], 2)
        c2 = self.generate_data([40, 5, 20], 1)
        X_test = np.concatenate((c1, c2))
        X_test = pd.DataFrame(X_test, columns=['f1', 'f2', 'f3'])

        scaled_X_train, scaled_X_test = self.test_model.scale_data(X_train, X_test)

        try:
            pd.testing.assert_frame_equal(scaled_X_train, X_train)
        except AssertionError:
            self.assertEqual(True, True)
            pass
        else:
            raise AssertionError

    ##########################################################################
    # OvAModel functions
    ##########################################################################

    def test_compute_probability_range_metrics(self):
        self.assertEqual(self.test_model.class_labels,
                         ['Unspecified CC', 'Unspecified Ia'])

        probs = np.array([[.9, .1],
                          [.1, .9]])
        y = [['Unspecified CC'],
             ['Unspecified Ia']]
        y_test = pd.DataFrame(y, columns=[TARGET_LABEL])
        label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
        results = np.hstack((probs, label_column))

        output = self.test_model.compute_probability_range_metrics(results, bin_size=.1)
        self.assertEqual(output['Unspecified Ia'][0][9], 1)

        # Test with larger bin size
        probs = np.array([[.95, .01],
                          [.01, .95]])
        y = [['Unspecified CC'],
             ['Unspecified Ia']]
        y_test = pd.DataFrame(y, columns=[TARGET_LABEL])
        label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
        results = np.hstack((probs, label_column))

        output = self.test_model.compute_probability_range_metrics(results, bin_size=.01)
        # true positive in range == 1
        self.assertEqual(output['Unspecified Ia'][0][94], 1)
        # total in range == 1
        self.assertEqual(output['Unspecified Ia'][1][94], 1)

    def test_compute_metrics(self):
        self.assertEqual(self.test_model.class_labels,
                         ['Unspecified CC', 'Unspecified Ia'])
        probs = np.array([[.9, .1],
                          [.9, .1],
                          [.1, .9],
                          [.1, .9]])
        y = [['Unspecified CC'],
             ['Unspecified Ia'],
             ['Unspecified CC'],
             ['Unspecified Ia']]
        y_test = pd.DataFrame(y, columns=[TARGET_LABEL])
        label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
        results = np.hstack((probs, label_column))
        class_metrics = self.test_model.compute_metrics(results)
        self.assertEqual(list(class_metrics['Unspecified CC'].values()), [1, 1, 1, 1])
        self.assertEqual(list(class_metrics['Unspecified Ia'].values()), [1, 1, 1, 1])

        # More complex case with 3 classes
        self.test_model.class_labels = ['A', 'B', 'C']
        self.assertEqual(self.test_model.class_labels, ['A', 'B', 'C'])
        probs = np.array([[.8, .1, .1],
                          [.8, .1, .1],
                          [.1, .7, .2],
                          [.1, .1, .8],
                          [.01, .09, .9]])
        y = [['A'],
             ['B'],
             ['B'],
             ['C'],
             ['C']]
        y_test = pd.DataFrame(y, columns=[TARGET_LABEL])
        label_column = y_test[TARGET_LABEL].values.reshape(-1, 1)
        results = np.hstack((probs, label_column))
        class_metrics = self.test_model.compute_metrics(results)
        self.assertEqual(list(class_metrics['A'].values()), [1, 1, 0, 3])
        self.assertEqual(list(class_metrics['B'].values()), [1, 0, 1, 3])
        self.assertEqual(list(class_metrics['C'].values()), [2, 0, 0, 3])

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

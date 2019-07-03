import pandas as pd
from thex_data.data_consts import class_to_subclass, PRED_LABEL


class ConditionalTest:

    def get_parent_prob(self, class_name, probabilities):
        """
        Recurse up through tree, getting parent prob until we find a valid one. For example, there may only be CC, II, II P in CC so we need to inherit the probability of CC.
        """
        if class_name == "TTypes":
            return 1
        elif self.get_parent(class_name) in probabilities:
            return probabilities[self.get_parent(class_name)]
        else:
            return self.get_parent_prob(self.get_parent(class_name), probabilities)

    def get_parent(self, class_name):
        """
        Get parent class name of this class in tree
        """
        for parent_class, subclasses in class_to_subclass.items():
            if class_name in subclasses:
                return parent_class

import pandas as pd

from thex_data.data_consts import TARGET_LABEL, class_to_subclass
from thex_data.data_clean import *

from models.network_model.subnetwork import SubNetwork


class NetworkTrain:

    def get_subnet_classes(self, classes, y):
        """
        Get list of classes this subnet is built to classify
        """
        # Because all classes passed in have the same parent, we only need to look
        # at the parent of this first one
        child_class = classes[0]
        for parent_class, subclasses in class_to_subclass.items():
            if child_class in subclasses:
                classes.append(parent_class)
                break
        # Only keep classes which exist in data
        unique_classes = self.get_mc_unique_classes(df=y)
        target_classes = list(set(unique_classes).intersection(classes))
        return target_classes

    def train(self):
        """
        Create series of SubNetworks connecting layers of class hierarchy
        """
        # Convert class labels to class vectors

        y = convert_class_vectors(
            self.y_train, self.class_labels, self.test_level)
        self.networks = {}
        for parent_class, subclasses in class_to_subclass.items():
            subnet_classes = self.get_subnet_classes(subclasses, self.y_train)
            net = SubNetwork(subnet_classes, self.X_train, y,
                             self.class_labels)
            self.networks[parent_class] = net

        return self.networks

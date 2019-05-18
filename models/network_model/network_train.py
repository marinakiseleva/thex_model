import pandas as pd

from thex_data.data_consts import TARGET_LABEL, class_to_subclass
from thex_data.data_clean import *

from models.network_model.subnetwork import SubNetwork


class NetworkTrain:

    def get_classes_by_level(self, classes):
        """
        List of classes (at all levels). Creating mapping from class level to classes at that level.
        """
        tree = init_tree()
        node_depths = assign_levels(tree, {}, tree.root, 1)
        class_levels = {}
        for class_name in classes:
            class_level = node_depths[class_name]
            if class_level in class_levels:
                class_levels[class_level].append(class_name)
            else:
                class_levels[class_level] = [class_name]
        return class_levels

    def train(self):
        """
        Create series of SubNetworks connecting layers of class hierarchy
        """
        # Convert class labels to class vectors
        y_train_vectors = convert_class_vectors(
            self.y_train, self.class_labels, self.test_level)

        classes = self.get_classes_by_level(self.class_labels)
        for level in classes.keys():
            cur_level_classes = classes[level]

            net = SubNetwork(level, cur_level_classes, self.X_train,
                             self.y_train, self.class_labels)

        return self.models

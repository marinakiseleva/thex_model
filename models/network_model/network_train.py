import pandas as pd

from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS, class_to_subclass
from thex_data.data_clean import *

from models.network_model.subnetwork import SubNetwork


class NetworkTrain:

    def get_subnet_classes(self, child_classes, y, parent_class):
        """
        Select classes by:
            1. add classes that exist in data and in child_classes
            2. add Undefined_parent class if a row contains no child classes, but does contain parent AND other rows also contain children. 

        :param classes: child classes
        :param parent_class: parent_class of classes we are considering
        :return: list of class names for this subnet; can be empty.
        """
        # 1. add classes that exist in data and in child_classes
        unique_classes = []
        add_undefined_parent = False  # Record undef parent labels
        for index, row in y.iterrows():
            labels = convert_str_to_list(row[TARGET_LABEL])
            unique_classes += list(set(child_classes).intersection(labels))
            if len(unique_classes) == 0 and parent_class in labels:
                add_undefined_parent = True

        """ 2. add Undefined_parent class if at least 1 row contained no child classes and another contained children (otherwise there is nothing to compare against)"""
        if len(unique_classes) >= 1 and add_undefined_parent:
            unique_classes.append(UNDEF_CLASS + parent_class)

        return (list(set(unique_classes)))

    def get_subnet_data(self, X, y, class_labels):
        """
        Filter features (X) and labels (y) to only include those rows that have a class within class_labels
        :param class_labels: List of classes (single class name, like the children in class_to_subclass)
        :return: X filtered down to same rows as y, which is filtered to only those samples that contain a class in class_labels. And relabeled as the index of the class in the list class_labels (need numeric label for later processing)
        """
        keep_indices = []  # Indices to keep in X
        labels = []  # Corresponding relabeled y
        for df_index, row in y.iterrows():
            cur_classes = convert_str_to_list(row[TARGET_LABEL])
            has_level_class = False
            for class_index, class_name in enumerate(class_labels):
                if class_name in cur_classes:
                    # if has_level_class:
                    #     print("Sample was already assigned " + str(class_labels[
                    #         labels[-1]]) + " and now it also wants " + str(class_name))
                    has_level_class = True
                    labels.append(class_labels.index(class_name))
                    keep_indices.append(df_index)
            if has_level_class == False:
                # Assign parent's undefined class if it has NO subclass.
                for class_index, class_name in enumerate(class_labels):
                    if UNDEF_CLASS in class_name and class_name[len(UNDEF_CLASS):] in cur_classes:
                        labels.append(class_labels.index(class_name))
                        keep_indices.append(df_index)
        # Filter X and y
        y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        X = X.loc[keep_indices].reset_index(drop=True)
        return X, y

    def train(self):
        """
        Create series of SubNetworks connecting layers of class hierarchy
        """
        for parent_class, subclasses in class_to_subclass.items():
            class_level = self.class_levels[parent_class]

            subnet_classes = self.get_subnet_classes(
                subclasses, self.y_train, parent_class)
            if len(subnet_classes) <= 1:
                # print("Do not need network for " + parent_class +
                #       " since it has no subclasses in dataset.")
                continue

            print("\n\nInitialzing SubNetwork for classes " + str(subnet_classes))
            X, y = self.get_subnet_data(self.X_train, self.y_train, subnet_classes)

            self.networks[parent_class] = SubNetwork(subnet_classes, X, y)

        return self.networks

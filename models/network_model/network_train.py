import pandas as pd

from thex_data.data_consts import TARGET_LABEL, UNDEF_CLASS, class_to_subclass
from thex_data.data_clean import *

from models.network_model.subnetwork import SubNetwork


class NetworkTrain:

    def get_labels_at_depth(self, y, depth, parent_class):
        """
        Get all unique labels in the dataset y that are at depth. If a sample has no label at this depth but it is the parent, the parent category is returned (to be used later as Undefined)
        """
        unique_classes = []
        for index, row in y.iterrows():
            labels = convert_str_to_list(row[TARGET_LABEL])
            has_level_class = False
            for label in labels:
                if label != '' and self.class_levels[label] == depth:
                    unique_classes.append(label)
                    has_level_class = True
            if has_level_class == False:
                for label in labels:
                    if label == parent_class:
                        unique_classes.append(label)

        return list(set(unique_classes))

    def get_subnet_classes(self, classes, y, depth):
        """
        Get list of class names that exist both in list of classes and in the data y
        :param depth: Tree depth of classes we are considering
        """
        # Because all classes passed in have the same parent, we only need to look
        # at the parent of this first one
        child_class = classes[0]
        for parent_class, subclasses in class_to_subclass.items():
            if child_class in subclasses:
                undefined_parent = UNDEF_CLASS + parent_class
                parent = parent_class
                break
        # Only keep classes which exist in data
        unique_classes = self.get_labels_at_depth(y, depth, parent)
        target_classes = list(set(unique_classes).intersection(classes))
        if parent in unique_classes:
            target_classes.append(undefined_parent)

        return target_classes

    def get_subnet_data(self, X, y, class_labels):
        """
        Filter features (X) and labels (y) to only include those rows that have a class within class_labels
        :param class_labels: List of classes (single class name, like the children in class_to_subclass)
        """
        keep_indices = []
        labels = []
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
                subclasses, self.y_train, class_level + 1)
            if len(subnet_classes) <= 1:
                print("Do not need network for " + parent_class +
                      " since it has no subclasses in dataset.")
                continue

            X, y = self.get_subnet_data(self.X_train, self.y_train, subnet_classes)

            self.networks[parent_class] = SubNetwork(subnet_classes, X, y)

        return self.networks

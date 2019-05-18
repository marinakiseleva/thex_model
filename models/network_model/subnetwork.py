import pandas as pd

from thex_data.data_consts import TARGET_LABEL
from thex_data.data_clean import convert_class_vectors


class SubNetwork:

    def __init__(self, level, classes, X, y, class_labels):
        """
        Neural network with first layer of length input_length, last layer of output_length, wher each output node corresponds to a class in classes
        """
        self.level = level
        self.input_length = len(list(X))
        self.output_length = len(classes)
        self.level_classes = classes

        print("Initialzing SubNetwork at level " +
              str(level) + "  for classes " + str(classes))

        y_train_vectors = convert_class_vectors(
            y, class_labels, None)

        # Relabel each class vector numeric label per class
        labels = []
        for df_index, row in y_train_vectors.iterrows():
            class_vector = row[TARGET_LABEL]
            class_label = -1
            for class_index, class_name in enumerate(class_labels):
                if class_name in self.level_classes and class_vector[class_index] == 1:
                    if class_label != -1:
                        raise ValueError(
                            "SubNetwork __init__: Each sample should only have 1 unique class per level of the hierarchy.")
                    class_label = class_name
            if class_label == -1:
                raise ValueError(
                    "SubNetwork __init__: Sample label was never set.")
            labels.append(class_label)
        class_vectors = pd.DataFrame(labels, columns=[TARGET_LABEL])

        print("Done relabeling ")
        print("original")
        print(y)
        print("classes to filter on ")
        print(self.level_classes)
        print(class_vectors)

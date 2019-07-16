import pandas as pd
# import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier
from thex_data.data_consts import TARGET_LABEL
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


class NetClassifier(BinaryClassifier):
    """
    Extension of abstract class for binary classifier.
    """

    def init_classifier(self, X, y):
        """
        Initialize the classifier by fitting the data to it.
        """
        self.input_length = len(list(X))
        self.model = self.get_best_model(X, y)
        return self.model

    def predict(self, X):
        """
        Get the probability of the positive class for each row in DataFrame X. Return probabilities as Numpy column.
        :param X: DataFrame of features
        Return 2D Numpy array as column with probability of class per row
        """
        return self.model.predict(x=X,  batch_size=12)

    def get_best_model(self, X, y):
        # Get weight of each sample by its class frequency
        # labeled_samples = pd.concat([X, y], axis=1)
        # class_weights = self.get_class_weights(labeled_samples)

        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.5)
        weights_train = self.get_sample_weights(y_train)
        weights_valid = self.get_sample_weights(y_valid)

        # Assign class weights
        class_weights = self.get_class_weights(y)

        es = EarlyStopping(monitor='val_loss',
                           mode='auto',
                           verbose=0,
                           min_delta=0,
                           patience=20,
                           restore_best_weights=True)

        model = Sequential()
        model.add(Dense(self.input_length,
                        input_dim=self.input_length,
                        kernel_initializer='normal',
                        activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(1,
                        kernel_initializer='normal',
                        activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        metrics = model.fit(x_train,  # X
                            y_train,  # y
                            epochs=150,
                            batch_size=32,
                            verbose=0,
                            class_weight=class_weights,
                            validation_data=(x_valid, y_valid),
                            callbacks=[es])

        print("\nFinished at epoch " + str(es.stopped_epoch))
        print("Validation loss: " + str(metrics.history['val_loss'][-1])
              + " , training loss: " + str(metrics.history['loss'][-1]))

        metrics = model.evaluate(x_valid.values, y_valid)
        print("Valiation " +
              str(model.metrics_names[0]) + " : " + str(round(metrics[0], 4)))

        if es.stopped_epoch == 0:
            print("Exit at max epochs: validation loss never increased.")
        return model

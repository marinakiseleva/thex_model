import pandas as pd
import numpy as np
from models.ensemble_models.ensemble_model.binary_classifier import BinaryClassifier
from thex_data.data_consts import TARGET_LABEL


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing


from keras.wrappers.scikit_learn import KerasClassifier
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
        self.model = self.init_model(X, y)
        return self.model

    def predict(self, X):
        """
        Get the probability of the positive class for each row in DataFrame X. Return probabilities as Numpy column.
        :param X: DataFrame of features
        Return 2D Numpy array as column with probability of class per row
        """
        return self.model.predict(x=X,  batch_size=12)

    def init_model(self, X, y):
        # Get weight of each sample by its class frequency
        # labeled_samples = pd.concat([X, y], axis=1)
        # class_weights = self.get_class_weights(labeled_samples)

        # mm_scaler = preprocessing.MinMaxScaler()
        # X_processed = mm_scaler.fit_transform(X)

        X_processed = preprocessing.normalize(X)
        x_train, x_valid, y_train, y_valid = train_test_split(
            X_processed, y, test_size=0.3)

        weights_train = self.get_sample_weights(y_train)
        weights_valid = self.get_sample_weights(y_valid)

        # Assign class weights
        class_weights = self.get_class_weights(y)

        epochs = 1500
        batch_size = 32
        verbosity = 0
        es = EarlyStopping(monitor='val_loss',
                           min_delta=0.0000001,
                           verbose=1,
                           patience=200,
                           restore_best_weights=True)

        # model = self.get_nn()
        model = self.get_best_model(batch_size=batch_size,
                                    epochs=epochs,
                                    x_valid=x_valid,
                                    y_valid=y_valid,
                                    val_sample_weights=weights_valid,
                                    x_train=x_train,
                                    y_train=y_train,
                                    train_sample_weights=weights_train,
                                    callbacks=[es],
                                    class_weights=class_weights)

        metrics = model.fit(x_train,  # X
                            y_train,  # y
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbosity,
                            class_weight=class_weights,
                            validation_data=(x_valid, y_valid),
                            callbacks=[es])

        print("\nFinished at epoch " + str(es.stopped_epoch))

        print("Validation loss: "
              + str(metrics.history['val_loss'][-1])
              + " , training loss: " + str(metrics.history['loss'][-1]))

        if es.stopped_epoch == 0:
            print("Exit at max epochs: validation loss never increased.")
        return model

    def get_nn(self):
        """
        Initialize neural network 
        """

        model = Sequential()
        num_neurons = int(self.input_length / 2)
        model.add(Dense(num_neurons,
                        input_dim=self.input_length,
                        kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1,
                        kernel_initializer='normal',
                        activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def get_best_model(self, batch_size, epochs, x_valid, y_valid, val_sample_weights, x_train, y_train, train_sample_weights, callbacks, class_weights):

        param_grid = {
            # 'learn_rate': [0.0001, 0.001],  # , 0.01
            # 'momentum': [0.5],  # usually between 0.5-0.9
            # 'decay': [0.2, 0.6],
            # 'nesterov': [True],  # False
            # , [88, 64, 16], [64, 38, 24, 12]], [88, 44, 32, 16]
            'layer_sizes': [[int(self.input_length / 2)],
                            [int(self.input_length)],
                            [92, 32],
                            [88, 64, 16],
                            [108, 64]],
            'input_length': [self.input_length]
        }

        grid_model = KerasClassifier(build_fn=create_model)

        # scoring_type = 'average_precision'
        scoring_type = 'brier_score_loss'
        grid = GridSearchCV(estimator=grid_model,
                            param_grid=param_grid,
                            n_jobs=-1,
                            verbose=0,
                            iid=False,  # Avg score across folds
                            cv=3,
                            scoring=scoring_type)

        keras_fit_params = {
            'shuffle': True,
            'callbacks': callbacks,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_data': (x_valid, y_valid),  # val_sample_weights
            #'sample_weight': train_sample_weights)
            'class_weight': class_weights,
            'verbose': 0

        }
        grid.fit(x_train, y_train, **keras_fit_params)

        model = create_model(
            # learn_rate=grid.best_params_['learn_rate'],
            # momentum=grid.best_params_['momentum'],
            # decay=grid.best_params_['decay'],
            # nesterov=grid.best_params_['nesterov'],
            layer_sizes=grid.best_params_['layer_sizes'],
            input_length=self.input_length)

        print("Best model parameters with scoring = " + scoring_type)
        print(grid.best_params_)

        return model


def create_model(layer_sizes=[60], input_length=0, learn_rate=0.1, momentum=0.1, decay=0.1, nesterov=False):
    model = Sequential()
    # Add fully connected layer
    model.add(Dense(layer_sizes[0], input_dim=input_length, activation='relu'))
    if len(layer_sizes) > 1:
        # Add more layers, as specified
        for layer_size in layer_sizes[1:-1]:
            model.add(Dense(layer_size, activation='relu'))

    # sgd = optimizers.SGD(lr=learn_rate, momentum=momentum,
    #                      decay=decay, nesterov=nesterov)
    model.add(Dense(1,
                    kernel_initializer='normal',
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

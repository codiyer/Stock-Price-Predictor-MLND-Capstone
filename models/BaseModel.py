from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import data_utils
import numpy as np


class BaseModel:

    def __init__(self, data, num_of_days,
                 train_valid_test_split_rate=0.8, train_valid_split_rate=0.5, normalization_factor=100):
        self.data = data
        self.normalized_data = data_utils.normalize_data(self.data, normalization_factor)
        self.features, self.output = data_utils.get_features_output(self.normalized_data)
        x_data, y_data = data_utils.process_data(self.features, self.output, num_of_days)
        x_train_valid, self.x_test, y_train_valid, self.y_test = data_utils.split_data(x_data,
                                                                                       y_data,
                                                                                       train_valid_test_split_rate)

        self.x_train, self.x_valid, self.y_train, self.y_valid = data_utils.split_data(x_train_valid,
                                                                                       y_train_valid,
                                                                                       train_valid_split_rate)

    def summary(self):
        return self.model.summary()

    def compile(self):
        return self.model.compile(loss='mean_squared_error',
                                  optimizer='adam',
                                  metrics=['mean_absolute_percentage_error', 'mean_absolute_error'])

    def fit(self, epochs):
        model_checkpointer = ModelCheckpoint(filepath=self.weights_file,
                                             verbose=1,
                                             save_best_only=True)
        early_stopping_checkpoint = EarlyStopping(patience=20)

        return self.model.fit(x=self.x_train,
                              y=self.y_train,
                              epochs=epochs,
                              callbacks=[model_checkpointer, early_stopping_checkpoint],
                              validation_data=(self.x_valid, self.y_valid))

    def load_weights(self):
        return self.model.load_weights(self.weights_file)

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred), mean_absolute_error(y_true,y_pred), self.mean_absolute_percentage_error(y_true, y_pred)

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        """
        Used for calculating mean absolute percentage error
        :param y_true: Actual y values
        :param y_pred: Predicted y values
        :return: Mean absolute percentage error metric value
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

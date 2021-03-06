from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,mean_absolute_error, mean_squared_error

from utils import data_utils

import numpy as np


class KNNModel:

    def __init__(self, data, num_of_days,
                 train_valid_test_split_rate=0.8, normalization_factor=100):
        """
        Constructor which preprocesses the data and fits the model.
        :param data: Data
        :param num_of_days: Number of days for which the data needs to be batched.
        :param train_valid_test_split_rate: Split ratio between training and testing set.
        :param normalization_factor: Factor used for normalization.
        """
        self.data = data
        self.normalized_data = data_utils.normalize_data(self.data, normalization_factor)
        self.features, self.output = data_utils.get_features_output(self.normalized_data)
        x_data, y_data = data_utils.process_data(self.features, self.output, num_of_days)
        self.x_train, self.x_test, self.y_train, self.y_test = data_utils.split_data(x_data,
                                                                                     y_data,
                                                                                     train_valid_test_split_rate)

        scorer = make_scorer(mean_absolute_error)
        knn = KNeighborsRegressor()
        k_range = list(range(5, 100))
        param_grid = dict(n_neighbors=k_range)
        self.grid_search = GridSearchCV(knn, param_grid, cv=5, scoring=scorer)
        self.fit()

    @staticmethod
    def reshape(x):
        """
        Reshapes 3D array into 2D array for usage in KNN. sklearn mandates to have 2D arrays for x_values.
        :return: Reshaped x values
        """
        n_samples, nx, ny = x.shape
        return x.reshape((n_samples, nx*ny))

    def fit(self):
        """
        Fits the model after reshaping the x_data using method:reshape
        """
        self.grid_search.fit(self.reshape(self.x_train), self.y_train)

    def predict(self, x):
        """
        Used to predict y values.
        """
        return self.grid_search.predict(self.reshape(x))

    def score(self, y_pred):
        """
        Used to calculate mean_absolute_error
        :param y_pred: predicted y values
        :return: mean_absolute_error
        """
        return mean_squared_error(self.y_test, y_pred), mean_absolute_error(self.y_test, y_pred), self.mean_absolute_percentage_error(self.y_test, y_pred)

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        """
        Used for calculating mean absolute percentage error
        :param y_true: Actual y values
        :param y_pred: Predicted y values
        :return: Mean absolute percentage error metric value
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

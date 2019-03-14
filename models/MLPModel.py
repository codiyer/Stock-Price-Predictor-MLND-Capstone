from keras.models import Sequential
from keras.layers import Dense, Flatten

from .BaseModel import BaseModel


class MLPModel(BaseModel):

    train_valid_test_split_rate = 0.8
    train_valid_split_rate = 0.5
    
    def __init__(self, ticker, data, num_of_days,
                 train_valid_test_split_rate=0.8, train_valid_split_rate=0.5, normalization_factor=100):
        self.ticker = ticker
        self.weights_file = 'weights/mlp/' + self.ticker + '.weights.best.hdf5'
        super(MLPModel, self).__init__(data, num_of_days,
                                       train_valid_test_split_rate, train_valid_split_rate, normalization_factor)
        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=self.x_train[0].shape))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(512))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1))


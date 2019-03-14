from keras.models import Sequential
from keras.layers import Dense, LSTM
from models.BaseModel import BaseModel


class LSTMModel(BaseModel):

    def __init__(self, ticker, data, num_of_days,
                 train_valid_test_split_rate=0.8, train_valid_split_rate=0.5, normalization_factor=100):
        self.ticker = ticker
        self.weights_file = 'weights/lstm/' + self.ticker + '.weights.best.hdf5'
        super(LSTMModel, self).__init__(data, num_of_days,
                                        train_valid_test_split_rate, train_valid_split_rate, normalization_factor)
        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=self.x_train[0].shape, return_sequences=True))
        self.model.add(LSTM(256))
        self.model.add(Dense(1))


from keras.models import Sequential
from keras.layers import Dense, LSTM
from models.BaseModel import BaseModel


class LSTMModel(BaseModel):

    def __init__(self, ticker, num_of_days):
        super(LSTMModel, self).__init__(ticker, num_of_days)
        self.model = Sequential()
        self.model.add(LSTM(256, input_shape=self.x_train[0].shape, return_sequences=True))
        self.model.add(LSTM(256))
        self.model.add(Dense(1))


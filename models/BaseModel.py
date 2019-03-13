from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import data_utils


class BaseModel:

    def __init__(self, ticker, num_of_days,
                 train_valid_test_split_rate=0.8, train_valid_split_rate=0.5, normalization_factor=100):
        self.ticker = ticker
        self.data = data_utils.get_data(ticker=ticker)
        self.normalized_data = data_utils.normalize_data(self.data, normalization_factor)
        self.features, self.output = data_utils.get_features_output(self.normalized_data)
        x_data, y_data = data_utils.process_data(self.features, self.output, num_of_days)
        x_train_valid, self.x_test, y_train_valid, self.y_test = data_utils.split_data(x_data,
                                                                                       y_data,
                                                                                       train_valid_test_split_rate)

        self.x_train, self.x_valid, self.y_train, self.y_valid = data_utils.split_data(x_train_valid,
                                                                                       y_train_valid,
                                                                                       train_valid_split_rate)
        self.weights_file = 'weights/' + self.ticker + '.weights.best.hdf5'

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

import numpy as np
import fix_yahoo_finance as yf


def get_data(ticker, start_date="1997-01-01", end_date="2017-12-31"):
    """
    The functions downloads the data from yahoo finance for the input ticker.
    Returns data as pandas dataframe
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data.round(2)


def get_features_output(data):
    """
    Splits the dataset into features and output.
    Returns features and output as pandas df.
    """
    output = data[["Close"]]
    features = data[["Open", "High", "Low", "Close"]]
    return features, output


def process_data(features, output, num_of_days):
    """
    Processes the data into x_values and y_values.
    x_values contains features for :num_of_days days. and y_values contains output for the next day.
    :param features:
    :param output:
    :param num_of_days: Number of days for which the data needs to be considered for analysis.
    :return: x_data and y_data as numpy arrays
    """
    x_data = list()
    y_data = list()
    for i in range(len(features) - num_of_days):
        x = np.array(features[i: i + num_of_days][:])
        x_data.append(x)
        y = np.array(output.values[i + num_of_days])
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


def split_data(x_data, y_data, split_rate):
    """
    Used for splitting the data into train, validation and testing sets. No shuffling of data.
    :param x_data: the x values
    :param y_data: the y values
    :param split_rate: the rate of spitting of data. Should be > 0 and < 1
    :return: x_values split into 2 parts and y_values split into 2 parts.
    """
    if not 0 < split_rate < 1:
        raise Exception("Split Rate should be between 0 and 1")

    split_number = int(len(x_data) * split_rate)
    x_train = x_data[:split_number]
    x_test = x_data[split_number:]
    y_train = y_data[:split_number]
    y_test = y_data[split_number:]
    return x_train, x_test, y_train, y_test


def normalize_data(data, normalization_factor):
    """
    Used for data normalization.
    :param data: data as Pandas dataframe
    :param normalization_factor: factor using which data would be normalized
    :return: Divides all values of the data using the normalization factor and returns normalized data frame
    """
    return data / normalization_factor


def preprocess(data, num_of_days):
    features, output = get_features_output(data)
    return process_data(features, output, num_of_days)

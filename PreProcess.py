import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

class AirQualityData:
    def __init__(self, file_path="AirQualityUCI.xlsx", target_column='PT08.S3(NOx)', look_back=15):
        self.data_df = pd.read_excel(file_path, index_col=0)
        self.data_df.drop(['Time'], 1, inplace=True)
        self.data_df['date'] = self.data_df.index
        self.data_df['date'] = pd.to_datetime(self.data_df['date'])
        self.target_column = target_column
        self.look_back = look_back
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self):
        target_values = self.data_df[self.target_column].values.reshape(-1, 1)
        scaled_target = self.min_max_scaler.fit_transform(target_values)
        return scaled_target

    def create_dataset(self, dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def split_data(self):
        train_size = int(len(self.dataset) * 0.6)
        valid_size = int((len(self.dataset) - train_size) / 2)
        test_size = int((len(self.dataset) - train_size) / 2)

        train_data = self.dataset[0:train_size]
        valid_data = self.dataset[train_size: train_size + valid_size]
        test_data = self.dataset[train_size + valid_size: len(self.dataset)]

        x_train, y_train = self.create_dataset(train_data, self.look_back)
        x_valid, y_valid = self.create_dataset(valid_data, self.look_back)
        x_test, y_test = self.create_dataset(test_data, self.look_back)

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def create_polynomial_features(self, degree, x_train, x_valid, x_test):
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)
        X_valid_mapped = poly.transform(x_valid)
        X_test_mapped = poly.transform(x_test)

        scaler = StandardScaler()
        X_train_mapped_scaled1 = scaler.fit_transform(X_train_mapped)
        X_train_mapped_scaled = np.reshape(X_train_mapped_scaled1, (X_train_mapped_scaled1.shape[0], 1,
                                                                   X_train_mapped_scaled1.shape[1]))

        X_valid_mapped_scaled1 = scaler.transform(X_valid_mapped)
        X_valid_mapped_scaled = np.reshape(X_valid_mapped_scaled1, (X_valid_mapped_scaled1.shape[0], 1,
                                                                   X_valid_mapped_scaled1.shape[1]))

        X_test_mapped_scaled1 = scaler.transform(X_test_mapped)
        X_test_mapped_scaled = np.reshape(X_test_mapped_scaled1, (X_test_mapped_scaled1.shape[0], 1,
                                                                 X_test_mapped_scaled1.shape[1]))

        return X_train_mapped_scaled, X_valid_mapped_scaled, X_test_mapped_scaled

import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

class NeuralNetworkModels:
    def __init__(self, models, X_train_mapped_scaled, y_train, X_cv_mapped_scaled, y_cv, X_test_mapped_scaled, y_test):
        self.models = models
        self.X_train_mapped_scaled = X_train_mapped_scaled
        self.y_train = y_train
        self.X_cv_mapped_scaled = X_cv_mapped_scaled
        self.y_cv = y_cv
        self.X_test_mapped_scaled = X_test_mapped_scaled
        self.y_test = y_test
        self.nn_train_mses = []
        self.nn_cv_mses = []
        self.nn_test_mses = []
        self.nn_train_times = []

    def train_models(self, epochs=20, batch_size=32, learning_rate=0.1):
        for model in self.models:
            start_time = time.time()

            # Setup the loss and optimizer
            model.compile(
                loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            )

            print(f"Training {model.name}...")

            # Train the model
            model.fit(
                self.X_train_mapped_scaled, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            end_time = time.time()
            training_time = end_time - start_time

            print(f"Done! Training time: {training_time:.2f} seconds\n")

            # Record the training MSEs
            yhat_train = model.predict(self.X_train_mapped_scaled)
            train_mse = mean_squared_error(self.y_train, yhat_train) / 2
            self.nn_train_mses.append(train_mse)

            # Record the cross-validation MSEs
            yhat_cv = model.predict(self.X_cv_mapped_scaled)
            cv_mse = mean_squared_error(self.y_cv, yhat_cv) / 2
            self.nn_cv_mses.append(cv_mse)

            # Record the test MSEs
            yhat_test = model.predict(self.X_test_mapped_scaled)
            test_mse = mean_squared_error(self.y_test, yhat_test) / 2
            self.nn_test_mses.append(test_mse)

            # Record the training time
            self.nn_train_times.append(training_time)

    def print_results(self):
        print("RESULTS:")
        for model_num in range(len(self.nn_train_mses)):
            print(
                f"Model {model_num+1}: Training MSE: {self.nn_train_mses[model_num]:f}, " +
                f"CV MSE: {self.nn_cv_mses[model_num]:f}, Test MSE: {self.nn_test_mses[model_num]:f}, " +
                f"Training Time: {self.nn_train_times[model_num]:.2f} seconds"
            )
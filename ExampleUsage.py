from Models import build_models
from PrePrcess import AirQualityData
from Train import NeuralNetworkModels
from Plotter import PlottingAndTable

# Example Usage:
air_quality_data = AirQualityData()
x_train, y_train, x_valid, y_valid, x_test, y_test = air_quality_data.split_data()
X_train_mapped_scaled, X_valid_mapped_scaled, X_test_mapped_scaled = air_quality_data.create_polynomial_features(
    degree=1, x_train=x_train, x_valid=x_valid, x_test=x_test
)

neural_network_models = NeuralNetworkModels(
    models=build_models(),
    X_train_mapped_scaled=X_train_mapped_scaled,
    y_train=y_train,
    X_cv_mapped_scaled=X_valid_mapped_scaled,
    y_cv=y_valid,
    X_test_mapped_scaled=X_test_mapped_scaled,
    y_test=y_test
)
neural_network_models.train_models()
neural_network_models.print_results()
nn_train_mses, nn_cv_mses, nn_test_mses, nn_train_times = neural_network_models.return_results()

plotting_and_table = PlottingAndTable(nn_train_mses, nn_cv_mses, nn_test_mses, nn_train_times)
plotting_and_table.plot_results()
plotting_and_table.create_table()

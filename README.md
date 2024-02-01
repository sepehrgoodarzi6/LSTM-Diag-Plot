# LSTM-Diag-Plot
OOP-based LSTM (Long short-term memory) with further diagnosis and analysis using plots and tables.


## Air Quality Prediction Package

### Overview

This Python package provides a comprehensive solution for air quality prediction, combining data preprocessing, neural network model construction, and result visualization. It is designed to facilitate the process of predicting air quality based on historical data.

### Key Components

#### AirQualityData Class

The `AirQualityData` class handles the preprocessing of air quality data, including reading data from an Excel file, handling time-related features, and scaling the dataset for model training.

#### NeuralNetworkModels Class

The `NeuralNetworkModels` class encapsulates the construction of various LSTM-based neural network models. These models are designed for air quality prediction, with configurable architecture and activation functions.

#### PlottingAndTable Class

The `PlottingAndTable` class focuses on result visualization, providing methods to plot training and validation errors over epochs and create informative tables summarizing model performance.

#### build_models Function

The `build_models` function is a utility function that simplifies the creation of multiple LSTM models with different architectures. It returns a list of pre-configured models ready for training.

### Example Usage

```python
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
    X_cv_mapped_scaled=X_cv_mapped_scaled,
    y_cv=y_valid,
    X_test_mapped_scaled=X_test_mapped_scaled,
    y_test=y_test
)
neural_network_models.train_models()
neural_network_models.print_results()
plotting_and_table = PlottingAndTable(nn_train_mses, nn_cv_mses, nn_test_mses, nn_train_times)
plotting_and_table.plot_results()
plotting_and_table.create_table()
```

### Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/air-quality-prediction.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the file named ExampleUsage.py to see the results.

### Contributions

Contributions are welcome! Feel free to open issues for bug reports or new feature requests. If you want to contribute, fork the repository and create a pull request.

### License

This package is distributed under the [MIT License](LICENSE).

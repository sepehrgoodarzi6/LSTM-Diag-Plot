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
# Example Usage
data_processor = AirQualityData("AirQualityUCI.xlsx")
data_processor.preprocess_data()

model_builder = NeuralNetworkModels(data_processor.x_train, data_processor.y_train)
nn_models = model_builder.build_models()

results_plotter = PlottingAndTable()
results_plotter.plot_results(nn_models, data_processor.x_train, data_processor.y_train, data_processor.x_valid, data_processor.y_valid)
results_plotter.create_table(nn_models, data_processor.x_train, data_processor.y_train, data_processor.x_valid, data_processor.y_valid)
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

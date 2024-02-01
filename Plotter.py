import matplotlib.pyplot as plt
import numpy as np

class PlottingAndTable:
    def __init__(self, nn_train_mses, nn_cv_mses, nn_test_mses, nn_train_times):
        self.nn_train_mses = nn_train_mses
        self.nn_cv_mses = nn_cv_mses
        self.nn_test_mses = nn_test_mses
        self.nn_train_times = nn_train_times

    def plot_results(self):
        x = [i for i in range(len(self.nn_train_mses))]
        plt.plot(x, self.nn_train_mses, label='train_mses')
        plt.plot(x, self.nn_cv_mses, label='cv_mses')
        plt.plot(x, self.nn_test_mses, label='test_mses')
        plt.legend()
        plt.savefig('results_mses.svg')
        plt.show()

    def create_table(self):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('off')

        table_data = [
            ["Model", "Training MSE", "CV MSE", "Test MSE", "Training Time (s)"]
        ]

        # Populate the table with results
        for model_num in range(len(self.nn_train_mses)):
            table_data.append([
                f"Model {model_num+1}",
                f"{self.nn_train_mses[model_num]:.6f}",
                f"{self.nn_cv_mses[model_num]:.6f}",
                f"{self.nn_test_mses[model_num]:.6f}",
                f"{self.nn_train_times[model_num]:.2f}"
            ])

        ax.table(cellText=table_data, loc='center', cellLoc='center', colLabels=None, cellColours=[['#f0f0f0']*5]*len(table_data))

        # Save the table as an SVG file
        fig.savefig('results_table.svg')
        plt.show()
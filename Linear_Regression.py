import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self):
        self.line = None
        self.alpha = None
        self.beta = None
        self.denominator = None

    def fit(self, data, targets, return_line=False):
        # Use your derivatives here to find the optimal solution
        # 'data' here is the data, and 'targets' will be the y = 5x + 3 you create in your main
        # Return the line of best fit
        dimensionality = (len(data), )
        if data.shape != dimensionality:
            raise Exception("Please ensure the data has dimensionality of 1")
        else:
            data = np.array(data)
            targets = np.array(targets)
            self.denominator = np.dot(data, data) - data.mean() * data.sum()
            self.alpha = (np.dot(data, targets) - targets.mean() * data.sum()) / self.denominator
            self.beta = (targets.mean() * np.dot(data, data) - data.mean() * np.dot(data, targets)) / self.denominator
            self.line = self.alpha * data + self.beta
            if return_line:
                return self.line

    def coefficients(self):
        if self.line is None:
            raise Exception("Coefficients cannot be calculated if the model hasn't been fit.")
        return self.alpha, self.beta

    def accuracy(self, targets):
        # Use this method to calculate the and return the R^2
        if self.line is None:
            raise Exception("Accuracy cannot be calculated if the model hasn't been fit.")
        numerator = ((targets - self.line) ** 2).sum()
        denominator = ((targets - targets.mean()) ** 2).sum()
        r_squared = 1 - numerator / denominator
        return r_squared

    def plot(self, data, targets):
        # Use this method to produce a plot of the line of best fit vs a scatter plot of the data
        if self.line is None:
            raise Exception("Plot cannot be constructed if the model hasn't been fit.")
        plt.scatter(data, targets, label="Data")
        plt.plot(data, self.line, color='red', label="Regression Line")
        plt.title("Regression VS Data")
        plt.xlabel("Data")
        plt.ylabel("Targets")
        plt.legend()
        plt.show()

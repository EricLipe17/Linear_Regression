from Linear_Regression import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv('D:\\Desktop\\Udemy_Classes\\LazyProgrammer\\machine_learning_examples-master\\linear_regression_class\\data_1d.csv')

data = np.random.normal(0, 1, 100000)

targets = 5 * data + 3

model = LinearRegression()
model.fit(df.iloc[:, 0], df.iloc[:, 1])
print("Model 1 coefficients: " + str(model.coefficients()))
print("Model 1 accuracy: " + str(model.accuracy(df.iloc[:, 1])))
model.plot(df.iloc[:, 0], df.iloc[:, 1])

model2 = LinearRegression()
model2.fit(data, targets)
print("\n\nModel 2 coefficients: " + str(model2.coefficients()))
print("Model 2 accuracy: " + str(model2.accuracy(targets)))
model2.plot(data, targets)

model3 = LinearRegression()
model3.fit(data, targets, gradient_descent=True, precision=10**-10)
print("\n\nModel 3 coefficients: " + str(model3.coefficients()))
print("Model 3 accuracy: " + str(model3.accuracy(targets)))
model3.plot(data, targets)

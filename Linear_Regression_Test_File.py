from Linear_Regression import LinearRegression
from Multiple_Regression import MultipleRegression
import pandas as pd
import numpy as np

# Importing some extra test data:
df = pd.read_csv('data_1d.csv')
df1 = pd.read_csv('data_2d.csv')

# Generating data as specified in assignment
data = np.random.normal(0, 1, 100)

# Generating targets
targets = 5 * data + 3

# Fitting model to imported data:
model = LinearRegression()
model.fit(df.iloc[:, 0], df.iloc[:, 1])
print("Model 1 coefficients: " + str(model.coefficients()))
print("Model 1 accuracy: " + str(model.accuracy()))
model.plot()

# Fitting model to generated data:
model2 = LinearRegression()
model2.fit(data, targets)
print("\n\nModel 2 coefficients: " + str(model2.coefficients()))
print("Model 2 accuracy: " + str(model2.accuracy()))
model2.plot()

# Fitting model to generated data again, but using gradient descent this time:
model3 = LinearRegression()
model3.fit(data, targets, gradient_descent=True, precision=10**-10)
print("\n\nModel 3 coefficients: " + str(model3.coefficients()))
print("Model 3 accuracy: " + str(model3.accuracy()))
model3.plot()

model4 = MultipleRegression()
model4.fit(df1.iloc[:, 0:2], df1.iloc[:, -1])
print("\n\nModel 4 coefficients: " + str(model4.coefficients()))
print("Model 4 accuracy: " + str(model4.accuracy()))
model4.plot()







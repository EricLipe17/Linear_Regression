from Linear_Regression import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv('D:\\Desktop\\Udemy_Classes\\LazyProgrammer\\machine_learning_examples-master\\linear_regression_class\\data_1d.csv')

data = np.random.normal(0, 1, 100)

targets = 5 * data + 3

model = LinearRegression()
model.fit(df.iloc[:, 0], df.iloc[:, 1])
print(model.accuracy(df.iloc[:, 1]))
model.plot(df.iloc[:, 0], df.iloc[:, 1])

model2 = LinearRegression()
model2.fit(data, targets)
print(model2.accuracy(targets))
model2.plot(data, targets)

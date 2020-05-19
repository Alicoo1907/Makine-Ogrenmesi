import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

myDataSet = pd.read_csv("Deneme.csv", sep=";")
x = myDataSet.iloc[:, [0, 1]].values
y = myDataSet.kilo.values.reshape(-1, 1)

ml_reg = LinearRegression()
ml_reg.fit(x, y)

b0 = ml_reg.predict([[0, 0]])
#18 yaş 170 cm X kilo
#14 yaş 150 cm X kilo
p_list = ml_reg.predict(np.array([[18, 170], [14, 150]]))
print(b0)
print(p_list)
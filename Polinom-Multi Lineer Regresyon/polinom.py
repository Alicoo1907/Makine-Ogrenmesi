import pandas as pd
import matplotlib.pyplot as myPlot
from sklearn.preprocessing import PolynomialFeatures

myData = pd.read_csv("pl.csv", sep=";")
x = myData.Yas.values.reshape(-1, 1)
y = myData.kilo.values.reshape(-1, 1)

myPlot.scatter(x, y)
myPlot.xlabel("Yas")
myPlot.ylabel("Kilo")
#myPlot.show()

# %%linear regression olsaydi
from sklearn.linear_model import LinearRegression

lReg = LinearRegression()
lReg.fit(x, y)
y_pr = lReg.predict(x)
myPlot.plot(x, y_pr, color="red", label="Linear Reg")
#myPlot.show()


polyReg = PolynomialFeatures(degree=2)
polyX = polyReg.fit_transform(x)
lReg2 = LinearRegression()
lReg2.fit(polyX, y)
y_pre2 = lReg2.predict(polyX)
myPlot.plot(x, y_pre2, color="blue", label="polyreg")
myPlot.legend()
myPlot.show()

myPlot.show()

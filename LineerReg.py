import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime

df = pd.read_csv("covid19Pik.csv", sep=";")
p = sns.regplot(df.Gun, df.VakaArtıs, ci=None, scatter_kws={'color': 'r', 's': 9});
p.set_title("CoVid Bitiş")
plt.xlim(0, 60)
plt.show()
x = df[["Gun"]]
y = df[["VakaArtıs"]]

reg = LinearRegression()
model = reg.fit(x, y)
# sabit
b0 = model.intercept_
# katsayı
b1 = model.coef_

# basari kosulu
# i = vaka sayisinin 5 ten küçük olduğu ilk gün
# a = vaka sayisi
a, i = 6, 54
while a > 5:
    i += 1
    a = int(model.predict([[i]]))

print(f'vaka sayısının 5 ten kucuk oldugu gun = {i} ')
print(f'{i}. gündeki toplam vaka sayısı: {a}')

# tespit edilen gündeki vaka sayisi negatif olursa
print(f'{i}.gündeki vaka sayısı negatif bir değer olduğu için covid19 '
      f'{i} ile {i - 1}. gün arasında bitecektir. ')
print(f'{i - 1}. gündeki toplam vaka sayısı: ', int(model.predict([[i - 1]])))

# baslangıc tarihi 10 mart 2020
start_date = "3/10/20"
date_1 = datetime.datetime.strptime(start_date, "%m/%d/%y")

# bitis tarihi başlangıç tarihinin i gün kadar sonrası
end_date = date_1 + datetime.timedelta(days=i)
print("Coronavirüs", end_date, "tarihinde bitecektir")

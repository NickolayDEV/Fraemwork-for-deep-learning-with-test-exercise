import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Layers import Sequential, Linear
from activationFuncs import Tanh,Sigmoid
from GradientDescent import  Stochastic_gradient_descent
from lossfunctions import MSELoss
from DLTensor  import Tensor
import matplotlib.pyplot as plt
data=pd.read_csv('data.csv')
print(data.columns)

#Удаление ненужного столбца.
del data['Unnamed: 32']
#Разделение данных на x и y и преобразование их в numpy arrays
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Закадируем категориальные данные
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Разделим данные на тестовые и тренировочные

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Проскалируем данные, чтобы их значения лежали в схожих пределах
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model= Sequential([Linear(30,100), Tanh(),Linear(100,1), Sigmoid()])
criteria=MSELoss()
optimization=Stochastic_gradient_descent(parameters=model.get_parameters(),alpha=0.0075)
X_train=Tensor(X_train,autograd=True)
y_train=Tensor(y_train,autograd=True)
for i in range(1000):
    #Предсказание
    pred=model.forward(X_train)
    # Расчет ошибки
    loss=criteria.forward(pred,y_train)
    # Обучение

    loss.backward(Tensor(np.ones_like(loss.data)))
    optimization.step()
    print(loss)
X_test=Tensor(X_test,autograd=True)
y_test=Tensor(y_test,autograd=True)

pred=model.forward(X_test)
plt.plot(np.arange(len(y_test.data)),y_test.data)
plt.plot(np.arange(len(y_test.data)),pred.data)
plt.show()

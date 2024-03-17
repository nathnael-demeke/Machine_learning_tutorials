import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier


iris_dataset = pd.read_csv("csv\\iris.data.csv")
X = iris_dataset.drop("class", axis=1)
Y = iris_dataset["class"]

train_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size=0.25)

scalar = StandardScaler()
scalar.fit(train_X)

train_X = scalar.transform(train_X)
test_X = scalar.transform(test_X)

k_model = KNeighborsClassifier(n_neighbors=5)
k_model.fit(train_X,train_Y)



prediction = k_model.predict([[5.1,3.5,1.4,0.2]])

print(prediction)
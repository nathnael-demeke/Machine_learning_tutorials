import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sbrn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('housing.csv')

dataset.dropna(inplace=True)
# print(dataset.info())

X = dataset.drop("median_house_value", axis=1)
Y = dataset["median_house_value"]
filename = "regression.sav"

X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.25)

# save_model = pickle.dump()

train_data = X_train.join(Y_train)
train_data['total_rooms'] = np.log(train_data["total_rooms"] + 1)
train_data['total_bedrooms'] = np.log(train_data["total_bedrooms"] + 1)
train_data['population'] = np.log(train_data["population"] + 1)
train_data['households'] = np.log(train_data["households"] + 1)

# train_data.hist(figsize=(15,8))
# plt.show()

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity))
train_data = train_data.drop("ocean_proximity", axis=1)
X_train, Y_train = train_data.drop(["median_house_value"], axis=1), train_data["median_house_value"]
print(train_data.info())
model = LinearRegression()
model.fit(X_train, Y_train)
#test data
test_data = X_test.join(Y_test)
test_data['total_rooms'] = np.log(test_data["total_rooms"] + 1)
test_data['total_bedrooms'] = np.log(test_data["total_bedrooms"] + 1)
test_data['population'] = np.log(test_data["population"] + 1)
test_data['households'] = np.log(test_data["households"] + 1)
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity))
test_data = test_data.drop("ocean_proximity", axis=1)
X_test, Y_test = test_data.drop("median_house_value", axis=1), test_data["median_house_value"]
# print(model.score(X_train, Y_train))
to_be_predicted = [[-121.26, 38.68 ,13.0 ,8.356319965828153 ,6.429719478039138, 7.575071699507561,6.434546518787453 ,5.2051 ,False ,True ,False ,False ,False]]
y_pred = model.predict(to_be_predicted)
print(np.array(X_test)[1])
print(y_pred)




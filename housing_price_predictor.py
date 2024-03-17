import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sbrn 
from sklearn.model_selection import train_test_split
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
print(train_data.info())



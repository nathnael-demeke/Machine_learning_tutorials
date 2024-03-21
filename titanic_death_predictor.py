import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
columns_dummies = ["sex", "parch", "home.dest", "fare", "sibsp","cabin", "embarked","pclass"]
def convert_dataset_to_dummies(dataset, columns):
    new_dataset = None
    for column in columns:
        try:
            dataset = pd.concat([dataset.drop(column, axis=1), pd.get_dummies(dataset[column]).add_prefix(f"{column} ")])
            new_dataset = dataset
        except Exception as e:
            print(f"column {column} " + str(e))
    
    return new_dataset 
    

dataset = pd.read_excel(r"csv\\titanic3.xls")
dataset = dataset.drop(["ticket","boat"], axis=1)
dataset = convert_dataset_to_dummies(dataset,columns_dummies)
X = dataset.drop("survived", axis=1)
Y = dataset.survived
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25)

linear_model = LinearRegression()
random_forest_model = RandomForestClassifier()

linear_model.fit(X_train,Y_train)
linear_model_prediction = linear_model.predict(X_test,Y_test)

random_forest_model.fit(X_train, Y_train)
random_forest_prediction = random_forest_model.predict(X_test,Y_test)


print("[Linear] ... " + str(linear_model_prediction))
print("[RandomForest] ... " + str(random_forest_prediction))

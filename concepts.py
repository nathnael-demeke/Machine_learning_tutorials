import numpy as np 
import pandas as pd 

dataset = pd.read_csv("adult.csv")

#one hot encoding
dataset.education.value_counts()
#getting the dummies
dummies = pd.get_dummies(dataset.education).add_prefix("education")
#removing the education column and adding the dummy column
dataset = pd.concat([dataset.drop("education", axis=1),  pd.get_dummies(dataset.education).add_prefix("education ", axis=1)])
dataset = pd.concat([dataset.drop("workclass", axis=1),  pd.get_dummies(dataset.workclass).add_prefix("workclass ", axis=1)])
dataset = pd.concat([dataset.drop('marital-status', axis=1),  pd.get_dummies(dataset["marital-status"]).add_prefix("marital-status  ", axis=1)])
dataset = pd.concat([dataset.drop("relationship", axis=1),  pd.get_dummies(dataset.relationship).add_prefix("relationship ", axis=1)])
dataset = pd.concat([dataset.drop("race", axis=1),  pd.get_dummies(dataset.race).add_prefix("reace ", axis=1)])
dataset = pd.concat([dataset.drop('native-country', axis=1),  pd.get_dummies(dataset["native-country"]).add_prefix("native-country ", axis=1)]) 
print(dataset.head(6))



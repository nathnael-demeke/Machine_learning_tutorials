import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv("adult.csv")
#one hot encoding
dataset.education.value_counts()
#getting the dummies
# dummies = pd.get_dummies(dataset.education).add_prefix("education")
#removing the education column and adding the dummy column
dataset = pd.concat([dataset.drop("occupation", axis=1),  pd.get_dummies(dataset.occupation).add_prefix("occupation ", axis=1)])
dataset = pd.concat([dataset.drop("workclass", axis=1),  pd.get_dummies(dataset.workclass).add_prefix("workclass ", axis=1)])
dataset = dataset.drop("education", axis=1)
dataset = pd.concat([dataset.drop('marital-status', axis=1),  pd.get_dummies(dataset["marital-status"]).add_prefix("marital-status  ", axis=1)])
dataset = pd.concat([dataset.drop("relationship", axis=1),  pd.get_dummies(dataset.relationship).add_prefix("relationship ", axis=1)])
dataset = pd.concat([dataset.drop("race", axis=1),  pd.get_dummies(dataset.race).add_prefix("reace ", axis=1)])
dataset = pd.concat([dataset.drop('native-country', axis=1),pd.get_dummies(dataset["native-country"]).add_prefix("native-country ", axis=1)])
dataset["gender"] = dataset["gender"].apply(lambda x: 1 if x == "Male" else 0)
dataset["income"] = dataset["income"].apply(lambda x: 1 if x == ">50l" else 0)

#plotting the data
# print(dataset)
# plt.figure(figsize=[15,10])
# sb.heatmap(dataset.corr(),annot=False,cmap="coolwarm")
# plt.show()
# sb._show_cmap()
print("finished reading the data")
correlations = dataset.corr()["income"].abs()
sorted_correlations = correlations.sort_values()
nums_of_colons_to_drop = int(0.8 * len(dataset.columns))
colns_to_drop = sorted_correlations.iloc[ :nums_of_colons_to_drop].index
dataset = dataset.drop(colns_to_drop,axis=1)

#training the ai model
# dataset = dataset.drop("fnlwgt", axis=1)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
print("finished splitting the dataset...")
train_X = train_dataset.drop("income", axis=1)
train_Y = train_dataset["income"]

test_X = test_dataset.drop("income", axis=1)
test_Y = test_dataset["income"]

#our model has been trained and now we are fitting it and testing it with our slitted test data
print("random forest started....")
forest = RandomForestClassifier()
forest.fit(train_X, train_Y)
result = forest.score(test_X, test_Y)

print("line 51: this is the test result " + str(result))

importances = dict(zip(forest.feature_names_in_,forest.feature_importances_))
importances = {k:v for k,v in sorted(importances.items(),key=lambda x: x[1], reverse=True)}

print(importances)
parma_grid = {
    "n_estimators": [50,100,250],
    "max_depth": [5,30,null],
    "min_samples_split": [2,4],
    "max_features": ['sqrt', 'log2']

}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parma_grid, verbose=10)

grid_search.fit(train_X, train_Y)
forest = grid_search.best_estimator_
print("this is the result of the grid search best estimator " + forest.score(test_X, test_Y))


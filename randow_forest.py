import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sbrn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import sklearn
heart_dataset = pd.read_csv(r"csv\\heart.csv")

#returns all of the nessary data about the column like max and min values the mean and also how many of them are there (counts)
heart_dataset.describe()
#returns True if the value inside of a column is null and False if it is not null
X = heart_dataset.drop("target", axis=1)
Y = heart_dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=99)

random_forest_model = RandomForestClassifier(criterion="gini", max_depth=8,min_samples_split=10,random_state=5)
random_forest_model.fit(X_train,Y_train)
#return the importances of a column 
importance = dict(zip(random_forest_model.feature_names_in_,random_forest_model.feature_importances_))

Y_prediction = random_forest_model.predict(X_test)


confusion = confusion_matrix(Y_test,Y_prediction)
accuracy  = accuracy_score(y_true=Y_test, y_pred=Y_prediction)

print("this is score " + str(random_forest_model.score(X_test,Y_test)))
print("this is the accuracy " + str(accuracy))

print("###################### predicting ######################")
prediction_dataset = pd.read_csv(r"csv\\predict_heart.csv")

y_pred = random_forest_model.predict(prediction_dataset)

print(list(y_pred))
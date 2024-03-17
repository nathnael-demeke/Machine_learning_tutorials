import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sbrn 

heart_dataset = pd.read_csv("heart.csv")
print(heart_dataset.describe())
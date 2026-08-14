import pandas as pd
import numpy as np
import pickle
from  sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


file=pd.read_csv('diabetes.csv')
x=file[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

y=file['Outcome'].values
x
import pandas as pd
import numpy as np
import pickle
from  sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


file=pd.read_csv('diabetes.csv')
x=file[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

y=file['Outcome'].values
x=np.asanyarray(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


file=pd.read_csv('heart.csv')
features_cols=['ca', 'age', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thal', 'thalach', 'oldpeak', 'slope', 'exang']

x=file[features_cols]
y=file['target'].values
y=np.array(y)



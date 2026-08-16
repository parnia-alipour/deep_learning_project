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


def preprocess_new(new):
    new=np.array(new,dtype=float)
    new[:,[0,3,8]]=np.log1p(new[:,[0,3,8]])
    return new


x=np.asanyarray(x)
x=preprocess_new(x)


scaler=preprocessing.StandardScaler().fit(x)
x_scaler=scaler.transform(x)

model=LogisticRegression(max_iter=10000,class_weight='balanced').fit(x_scaler,y)

with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('heart_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


file=pd.read_csv('heart.csv')
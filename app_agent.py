import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model

API=FastAPI()

model=load_model("HPLC.keras")
with open("HPLC.keras","rb") as f:
    Artifacts=pickle.load(f)
scaler=Artifacts["scaler"]
lb2=Artifacts["label_encoder_age"]
lb3=Artifacts["label_encoder_workclass"]
lb4=Artifacts["label_encoder_jaundice"]
lb5=Artifacts["label_encoder_gender"]
threshold=Artifacts["best_threshold"]


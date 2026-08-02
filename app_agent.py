import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pyexpat import features
from tensorflow.keras.models import load_model

API=FastAPI()

model=load_model("HPLC.keras")
with open("thalassemia_artifacts.pkl","rb") as f:
    Artifacts=pickle.load(f)
scaler=Artifacts["scaler"]
lb2=Artifacts["label_encoder_age"]
lb3=Artifacts["label_encoder_weakness"]
lb4=Artifacts["label_encoder_jaundice"]
lb5=Artifacts["label_encoder_gender"]
threshold=Artifacts["best_threshold"]

class Prediction(BaseModel):
    Age: str
    Gender: str
    HbA0: float
    HbA2: float
    HbF: float
    S_Window:float
    RBC:float
    HB:float
    MCV:float
    MCH: float
    MCHC: float
    RDWcv:float
    Weekness: str
    Jaundice:str


@app.post("/predict/thalassemia")
def predict(data:Prediction):
    age_enc=lb2.transform([data.Age])[0]
    gender_enc=lb5.transform([data.Gender])[0]
    weakness_enc=lb3.transform([data.Weekness])[0]
    jaundice_enc=lb4.transform([data.Jaundice])[0]

    features=np.array([[age_enc, gender_enc, data.HbA0, data.HbA2, data.HbF,
        data.S_Window, data.RBC, data.HB, data.MCV, data.MCH,
        data.MCHC, data.RDWcv, weakness_enc, jaundice_enc
    ]])

    features_scaled=scaler.transform(features)
    prob=float(model.predict(features_scaled)[0][0])
    diagnosis="sick" if prob>threshold else "healthy"

    return{
        "diagnosis":diagnosis,
        "sick_probability_percent":round(prob*100,2),
    "healthy_probability_percent":round((prob*100,2))
}
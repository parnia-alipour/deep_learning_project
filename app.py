import pickle
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import gradio as gr
import spaces
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field


app=FastAPI(
    title="Medical Prediction API",
    description="Thalassemia + Diabetes prediction service, powered by the developer's own trained models.",
    version="1.0.0",
)



thalassemia_model=load_model("HPLC.keras")
with open("thalassemia_artifacts.pkl","rb") as f:
    thal_artifacts = pickle.load(f)
thal_scaler=thal_artifacts["scaler"]
thal_le_age=thal_artifacts["label_encoder_age"]
thal_le_weakness=thal_artifacts["label_encoder_weakness"]
thal_le_jaundice=thal_artifacts["label_encoder_jaundice"]
thal_le_gender=thal_artifacts["label_encoder_gender"]
thal_threshold=thal_artifacts["best_threshold"]

diabetes_model=None

try:
    with open("diabetes_model.pkl", "rb") as f:
        diabetes_model = pickle.load(f)
except FileNotFoundError:
    diabetes_model = None


class ThalassemiaInput(BaseModel):
    Age:str =Field(..., description='format: "25 Yrs 3 month"')
    Gender:str
    HbA0:float
    HbA2:float
    HbF:float
    S_Window:float
    RBC: float
    HB: float
    MCV: float
    MCH: float
    MCHC: float
    RDWcv: float
    Weekness: str
    Jaundice: str


class DiabetesInput(BaseModel):
    Pregnancies:float
    Glucose:float
    BloodPressure:float
    SkinThickness:float
    Insulin:float
    BMI:float
    DiabetesPedigreeFunction: float
    Age:float

@app.get("/")
def health_check():
    return {"status": "ok","diabetes_model_loaded": diabetes_model is not None}

@app.post("/predict/thalassemia")
def predict_thalassemia(data: ThalassemiaInput):
    try:
        age_enc=thal_le_age.transform([data.Age])[0]
        gender_enc=thal_le_gender.transform([data.Gender])[0]
        weakness_enc=thal_le_weakness.transform([data.Weekness])[0]
        jaundice_enc=thal_le_jaundice.transform([data.Jaundice])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value: {e}")


    features=np.array([[age_enc, gender_enc, data.HbA0, data.HbA2, data.HbF,
        data.S_Window, data.RBC, data.HB, data.MCV, data.MCH,
        data.MCHC, data.RDWcv, weakness_enc, jaundice_enc]])

    features_scaled=thal_scaler.transform(features)
    prob=float(thalassemia_model.predict(features_scaled,verbose=0)[0][0])
    diagnosis="sick" if prob > thal_threshold else "healthy"

    return {
        "diagnosis": diagnosis,
        "sick_probability_percent":round(prob * 100, 2),
        "healthy_probability_percent": round((1 - prob) * 100, 2),
    }

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    if diabetes_model is None:
        raise HTTPException(status_code=503,detail="The diabetes model hasn't been uploaded to the server yet (diabetes_model.pkl not found)")

    features=np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
        data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]])


    pred=int(diabetes_model.predict(features)[0])

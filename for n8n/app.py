import pickle
import numpy as np
from ipykernel.heartbeat import Heartbeat
from pydantic import BaseModel
from pyexpat import features
from tensorflow.keras.models import load_model

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
        diabetes_model=pickle.load(f)
except FileNotFoundError:
    diabetes_model=None


heart_model=None
scaler_heart=None
try:
    with open("heart_model.pkl", "rb") as f:
        heart_model=pickle.load(f)
    with open("heart_sclaer.pkl", "rb") as f:
        scaler_heart=pickle.load(f)
except FileNotFoundError:
    heart_model=None
    scaler_heart=None

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




class HeartInput(BaseModel):
    class HeartInput(BaseModel):
        ca:float=Field(...,description="number of major vessels colored by fluoroscopy (0-3)")
        age:float
        trestbps:float=Field(...,description="resting blood pressure")
        chol:float=Field(...,description="serum cholesterol (mg/dl)")
        fbs:float=Field(...,description="fasting blood sugar>120:1=yes,0=no")
        restecg:float=Field(...,description="resting ECG results (0-2)")
        thal:float=Field(...,description="thallium stress test result (1-3)")
        thalach:float=Field(...,description="maximum heart rate achieved")
        oldpeak:float=Field(...,description="ST depression induced by exercise")
        slope:float=Field(...,description="slope of peak exercise ST segment (0-2)")
        exang:float=Field(...,description="exercise-induced angina: 1=yes, 0=no")





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
    proba=diabetes_model.predict_proba(features)[0]
    sick_prob=float(proba[1])

    return {
        "diagnosis":"sick" if pred==1 else "healthy",
        "sick_probability_percent":round(sick_prob * 100, 2),
        "healthy_probability_percent": round((1 - sick_prob) * 100, 2),
    }



@app.post("/predict/heart")
def predict_heart(data: HeartInput ):
    if heart_model is None or  scaler_heart is None:
        raise  HTTPException(
            status_code=503,
            detail="The heart model hasn't been uploaded to the server yet (heart_model.pkl not found)"
        )



    features=np.array([[data.ca,data.age,data.trestbps,data.chol, data.fbs,
        data.restecg, data.thal,data.thalach, data.oldpeak,
        data.slope, data.exang]])

    features[:,[0,3,8]]=np.log1p(features[:,[0,3,8]])
    features_scaled=scaler_heart.transform(features)


    pred=int(heart_model.predict(features_scaled)[0])
    proba=heart_model.predict_proba(features_scaled)[0]

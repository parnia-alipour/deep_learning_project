import pickle
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import gradio as gr
import spaces


model=load_model("HPLC.keras")
with open("thalassemia_artifacts.pkl","rb") as f:
    Artifacts=pickle.load(f)
scaler=Artifacts["scaler"]
lb2=Artifacts["label_encoder_age"]
lb3=Artifacts["label_encoder_weakness"]
lb4=Artifacts["label_encoder_jaundice"]
lb5=Artifacts["label_encoder_gender"]
threshold=Artifacts["best_threshold"]

def predict(Age,Gender,HbA0,HbA2,HbF,S_Window,RBC,HB,MCV,MCH,MCHC,RDWcv,Weekness,Jaundice):
    age_enc=lb2.transform([Age])[0]
    gender_enc=lb5.transform([Gender])[0]
    weakness_enc=lb3.transform([Weekness])[0]
    jaundice_enc=lb4.transform([Jaundice])[0]

    features=np.array([[age_enc,gender_enc, HbA0,HbA2,HbF,
        S_Window, RBC, HB,MCV, MCH,
        MCHC,RDWcv, weakness_enc, jaundice_enc
    ]])

    features_scaled=scaler.transform(features)
    prob=float(model.predict(features_scaled)[0][0])
    diagnosis="sick" if prob>threshold else "healthy"

    return{
        "diagnosis":diagnosis,
        "sick_probability_percent":round(prob*100,2),
    "healthy_probability_percent":round((1-prob)*100,2)
}

demo=gr.Interface(fn=predict,inputs=[
        gr.Textbox(label="Age"),
        gr.Textbox(label="Gender"),
        gr.Number(label="HbA0"),
        gr.Number(label="HbA2"),
        gr.Number(label="HbF"),
        gr.Number(label="S_Window"),
        gr.Number(label="RBC"),
        gr.Number(label="HB"),
        gr.Number(label="MCV"),
        gr.Number(label="MCH"),
        gr.Number(label="MCHC"),
        gr.Number(label="RDWcv"),
        gr.Textbox(label="Weekness"),
        gr.Textbox(label="Jaundice"),
    ],outputs="json")

demo.launch()
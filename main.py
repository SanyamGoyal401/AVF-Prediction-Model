from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

class InputData(BaseModel):
    Age: float = Field(alias = 'Age')
    Gender: str = Field(alias = 'Gender')
    Comorbidity: str = Field(alias = 'Comorbidity')
    Smoking: str = Field(alias = 'Smoking')
    Alcohol: str = Field(alias = 'Alcohol')
    Basic_Disease: str = Field(alias='Basic Disease')
    AVF: str = Field(alias='AVF')
    Surgeon_Experience: str = Field(alias='Surgeon Experience')
    Venous_Course: str = Field(alias='Venous Course')
    Venous_Caliber: str = Field(alias='Venous Caliber')
    Venous_Stenosis: str = Field(alias='Venous Stenosis')
    Type_of_Anastomosis: str = Field(alias='Type of Anastomosis')
    Arterial_Wall: str = Field(alias='Arterial Wall')
    Vein_Diameter: float = Field(alias='Vein Diameter')
    Artery_Diameter: float = Field(alias='Artery Diameter')
    Artery_to_Vein_Distance: float = Field(alias='Artery to Vein Distance')

class PredictionResponse(BaseModel):
    AVF_Group: dict
    Post_Op_Complications: dict
    Post_Op_Thrill: dict
    Post_Op_Bruit: dict

app = FastAPI(title="AVF Prediction Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=False,
    allow_headers=["*"]
)

# Load models and encoders
avf_group_model = joblib.load('joblib/avf_group_model.joblib')
avf_group_encoder = joblib.load('joblib/avf_group_encoder.joblib')
complications_model = joblib.load('joblib/complications_model.joblib')
complications_encoder = joblib.load('joblib/complications_encoder.joblib')
thrill_model = joblib.load('joblib/thrill_model.joblib')
thrill_encoder = joblib.load('joblib/thrill_encoder.joblib')
bruit_model = joblib.load('joblib/bruit_model.joblib')
bruit_encoder = joblib.load('joblib/bruit_encoder.joblib')


@app.get("/hello")
async def root():
    return {"Message": "Server is live"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.model_dump(by_alias=True)])
        
        # Predict AVF Group
        avf_group_pred_encoded = avf_group_model.predict(input_df)[0]
        avf_group_pred = avf_group_encoder.inverse_transform([avf_group_pred_encoded])[0]
        avf_group_proba = avf_group_model.predict_proba(input_df)[0]
        avf_group_prob_dict = dict(zip(avf_group_encoder.classes_, avf_group_proba))
        
        # Add predicted AVF Group to input
        input_df['AVF Group'] = avf_group_pred
        
        # Predict subsequent outcomes
        predictions = {}
        
        # Complications
        comp_pred_encoded = complications_model.predict(input_df)[0]
        comp_pred = complications_encoder.inverse_transform([comp_pred_encoded])[0]
        comp_proba = complications_model.predict_proba(input_df)[0]
        comp_prob_dict = dict(zip(complications_encoder.classes_, comp_proba))
        
        # Thrill
        thrill_pred_encoded = thrill_model.predict(input_df)[0]
        thrill_pred = thrill_encoder.inverse_transform([thrill_pred_encoded])[0]
        thrill_proba = thrill_model.predict_proba(input_df)[0]
        thrill_prob_dict = dict(zip(thrill_encoder.classes_, thrill_proba))
        
        # Bruit
        bruit_pred_encoded = bruit_model.predict(input_df)[0]
        bruit_pred = bruit_encoder.inverse_transform([bruit_pred_encoded])[0]
        bruit_proba = bruit_model.predict_proba(input_df)[0]
        bruit_prob_dict = dict(zip(bruit_encoder.classes_, bruit_proba))
        
        return PredictionResponse(
            AVF_Group=avf_group_prob_dict,
            Post_Op_Complications=comp_prob_dict,
            Post_Op_Thrill=thrill_prob_dict,
            Post_Op_Bruit=bruit_prob_dict
        )
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

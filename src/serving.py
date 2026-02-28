"""
serving.py
API REST para servir el modelo de predicción de Churn.
Endpoint: POST /predict
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ─────────────────────────────────────────
# CARGAR MODELO Y SCALER
# ─────────────────────────────────────────
model  = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Columnas usadas en el entrenamiento (mismo orden)
FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# ─────────────────────────────────────────
# ESQUEMA DE ENTRADA
# ─────────────────────────────────────────
class CustomerData(BaseModel):
    gender: int                                     # 1=Male, 0=Female
    SeniorCitizen: int                              # 1=Yes, 0=No
    Partner: int                                    # 1=Yes, 0=No
    Dependents: int                                 # 1=Yes, 0=No
    tenure: int                                     # Meses como cliente
    PhoneService: int                               # 1=Yes, 0=No
    PaperlessBilling: int                           # 1=Yes, 0=No
    MonthlyCharges: float                           # Cargo mensual
    TotalCharges: float                             # Cargo total
    MultipleLines_No_phone_service: int             # 1=Yes, 0=No
    MultipleLines_Yes: int                          # 1=Yes, 0=No
    InternetService_Fiber_optic: int                # 1=Yes, 0=No
    InternetService_No: int                         # 1=Yes, 0=No
    OnlineSecurity_No_internet_service: int         # 1=Yes, 0=No
    OnlineSecurity_Yes: int                         # 1=Yes, 0=No
    OnlineBackup_No_internet_service: int           # 1=Yes, 0=No
    OnlineBackup_Yes: int                           # 1=Yes, 0=No
    DeviceProtection_No_internet_service: int       # 1=Yes, 0=No
    DeviceProtection_Yes: int                       # 1=Yes, 0=No
    TechSupport_No_internet_service: int            # 1=Yes, 0=No
    TechSupport_Yes: int                            # 1=Yes, 0=No
    StreamingTV_No_internet_service: int            # 1=Yes, 0=No
    StreamingTV_Yes: int                            # 1=Yes, 0=No
    StreamingMovies_No_internet_service: int        # 1=Yes, 0=No
    StreamingMovies_Yes: int                        # 1=Yes, 0=No
    Contract_One_year: int                          # 1=Yes, 0=No
    Contract_Two_year: int                          # 1=Yes, 0=No
    PaymentMethod_Credit_card_automatic: int        # 1=Yes, 0=No
    PaymentMethod_Electronic_check: int             # 1=Yes, 0=No
    PaymentMethod_Mailed_check: int                 # 1=Yes, 0=No


# ─────────────────────────────────────────
# APP FASTAPI
# ─────────────────────────────────────────
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predecir si un cliente abandonará el servicio",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "✅ Telco Churn Prediction API funcionando correctamente"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):

    # Convertir input a DataFrame
    input_dict = {
        'gender': customer.gender,
        'SeniorCitizen': customer.SeniorCitizen,
        'Partner': customer.Partner,
        'Dependents': customer.Dependents,
        'tenure': customer.tenure,
        'PhoneService': customer.PhoneService,
        'PaperlessBilling': customer.PaperlessBilling,
        'MonthlyCharges': customer.MonthlyCharges,
        'TotalCharges': customer.TotalCharges,
        'MultipleLines_No phone service': customer.MultipleLines_No_phone_service,
        'MultipleLines_Yes': customer.MultipleLines_Yes,
        'InternetService_Fiber optic': customer.InternetService_Fiber_optic,
        'InternetService_No': customer.InternetService_No,
        'OnlineSecurity_No internet service': customer.OnlineSecurity_No_internet_service,
        'OnlineSecurity_Yes': customer.OnlineSecurity_Yes,
        'OnlineBackup_No internet service': customer.OnlineBackup_No_internet_service,
        'OnlineBackup_Yes': customer.OnlineBackup_Yes,
        'DeviceProtection_No internet service': customer.DeviceProtection_No_internet_service,
        'DeviceProtection_Yes': customer.DeviceProtection_Yes,
        'TechSupport_No internet service': customer.TechSupport_No_internet_service,
        'TechSupport_Yes': customer.TechSupport_Yes,
        'StreamingTV_No internet service': customer.StreamingTV_No_internet_service,
        'StreamingTV_Yes': customer.StreamingTV_Yes,
        'StreamingMovies_No internet service': customer.StreamingMovies_No_internet_service,
        'StreamingMovies_Yes': customer.StreamingMovies_Yes,
        'Contract_One year': customer.Contract_One_year,
        'Contract_Two year': customer.Contract_Two_year,
        'PaymentMethod_Credit card (automatic)': customer.PaymentMethod_Credit_card_automatic,
        'PaymentMethod_Electronic check': customer.PaymentMethod_Electronic_check,
        'PaymentMethod_Mailed check': customer.PaymentMethod_Mailed_check,
    }

    df_input = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    # Escalar y predecir
    X_scaled    = scaler.transform(df_input)
    prediction  = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_label":      "Sí abandona" if prediction == 1 else "No abandona",
        "churn_probability": round(float(probability), 4)
    }
"""
data_preparation.py
Prepara el dataset raw de Telco Customer Churn para entrenamiento.
Input:  data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
Output: data/training/telco_churn_training.csv
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────
RAW_PATH      = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRAINING_PATH = "data/training/telco_churn_training.csv"


# ─────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset raw."""
    df = pd.read_csv(path)
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza general del dataset."""

    # Eliminar columna ID (no aporta al modelo)
    df = df.drop(columns=['customerID'])

    # Convertir TotalCharges a numérico (viene como texto)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Eliminar filas con valores nulos (son muy pocas, ~11 filas)
    nulos_antes = df.isnull().sum().sum()
    df = df.dropna()
    print(f"✅ Limpieza: {nulos_antes} valores nulos eliminados")
    print(f"✅ Dataset tras limpieza: {df.shape[0]} filas")

    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas binarias Yes/No y Female/Male a 1/0."""

    binary_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']

    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Género
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    print(f"✅ Columnas binarias codificadas")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica One-Hot Encoding a columnas categóricas multiclase."""

    categorical_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"✅ One-Hot Encoding aplicado")
    print(f"✅ Dataset final: {df.shape[0]} filas, {df.shape[1]} columnas")

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Guarda el dataset procesado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Dataset guardado en: {path}")


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def run_pipeline():
    print("\n🚀 Iniciando pipeline de preparación de datos...\n")

    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    save_data(df, TRAINING_PATH)

    print("\n✅ Pipeline completado exitosamente")
    print(f"📋 Columnas del dataset de entrenamiento:")
    for col in df.columns.tolist():
        print(f"   - {col}")


if __name__ == "__main__":
    run_pipeline()
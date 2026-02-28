"""
train.py
Entrena un modelo de clasificación para predecir Churn.
Input:  data/training/telco_churn_training.csv
Output: models/churn_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────
TRAINING_PATH = "data/training/telco_churn_training.csv"
MODELS_DIR    = "models"
MODEL_PATH    = "models/churn_model.pkl"
SCALER_PATH   = "models/scaler.pkl"
METRICS_PATH  = "reports/metrics.json"


# ─────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────

def load_data(path: str):
    """Carga el dataset de entrenamiento y separa features y target."""
    df = pd.read_csv(path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    print(f"✅ Dataset cargado: {X.shape[0]} filas, {X.shape[1]} features")
    return X, y


def split_data(X, y):
    """Divide en train y test (80/20)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """Escala las variables numéricas."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"✅ Datos escalados")
    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, y_train):
    """Entrena múltiples modelos y retorna el mejor."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"✅ Modelo entrenado: {name}")

    return trained


def evaluate_models(trained_models, X_test, y_test):
    """Evalúa cada modelo y retorna el mejor por Recall."""
    print("\n📊 RESULTADOS DE EVALUACIÓN:")
    print("=" * 50)

    best_model_name = None
    best_recall     = 0
    best_model      = None
    all_metrics     = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        recall  = recall_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"\n🔹 {name}")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")

        all_metrics[name] = {
            "accuracy": round(acc, 4),
            "recall":   round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc":  round(roc_auc, 4)
        }

        if recall > best_recall:
            best_recall     = recall
            best_model_name = name
            best_model      = model

    print(f"\n🏆 Mejor modelo: {best_model_name} (Recall: {best_recall:.4f})")
    return best_model, best_model_name, all_metrics


def save_confusion_matrix(model, X_test, y_test, model_name):
    """Guarda la matriz de confusión como imagen."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'], ax=ax)
    ax.set_title(f'Matriz de Confusión - {model_name}', fontweight='bold')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')
    plt.tight_layout()
    plt.savefig('reports/05_confusion_matrix.png', dpi=150)
    plt.close()
    print("✅ Matriz de confusión guardada en reports/")


def save_model(model, scaler, metrics):
    """Serializa y guarda el modelo, scaler y métricas."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Modelo guardado en:  {MODEL_PATH}")
    print(f"✅ Scaler guardado en:  {SCALER_PATH}")
    print(f"✅ Métricas guardadas en: {METRICS_PATH}")


# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def run_pipeline():
    print("\n🚀 Iniciando pipeline de entrenamiento...\n")

    X, y                              = load_data(TRAINING_PATH)
    X_train, X_test, y_train, y_test  = split_data(X, y)
    X_train_s, X_test_s, scaler       = scale_data(X_train, X_test)
    trained_models                    = train_models(X_train_s, y_train)
    best_model, best_name, metrics    = evaluate_models(trained_models, X_test_s, y_test)

    save_confusion_matrix(best_model, X_test_s, y_test, best_name)
    save_model(best_model, scaler, metrics)

    print("\n✅ Pipeline de entrenamiento completado exitosamente")
    print(f"🏆 Modelo champion: {best_name}")


if __name__ == "__main__":
    run_pipeline()
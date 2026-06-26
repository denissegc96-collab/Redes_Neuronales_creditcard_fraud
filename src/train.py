"""
Detección de fraude en tarjetas de crédito con una Red Neuronal (TensorFlow/Keras).

Tarea de Redes Neuronales Artificiales — Dr. Jorge Velazquez-Castro.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Este script entrena VARIOS modelos (varios entrenamientos) y registra para cada uno:
  - Curvas de loss y accuracy (entrenamiento y validación)  ->  results/*.png
  - Métricas y artefactos en MLflow                          ->  ./mlruns
  - Un resumen comparativo                                   ->  results/resumen.csv

Uso:
    python src/train.py --data archive/creditcard.csv --epochs 40

Para abrir el panel de MLflow tras entrenar:
    mlflow ui            # luego abrir http://127.0.0.1:5000
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sin ventana (para guardar PNG en cualquier entorno)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import mlflow

SEED = 42
RESULTS_DIR = "results"


# --------------------------------------------------------------------------- #
# 1) Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# 2) Carga y preprocesamiento de los datos
# --------------------------------------------------------------------------- #
def load_and_prepare(path: str):
    """Carga el CSV, escala 'Amount' y 'Time' y separa en train/val/test
    de forma estratificada (manteniendo la proporción de fraude)."""
    df = pd.read_csv(path)
    print(f"Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")

    fraude = int(df["Class"].sum())
    print(
        f"Transacciones fraudulentas: {fraude:,} "
        f"({100 * fraude / len(df):.3f} % del total)  ->  problema MUY desbalanceado"
    )

    # 'V1'..'V28' ya están en una escala comparable (salida de un PCA).
    # 'Amount' y 'Time' no, y tienen valores extremos -> RobustScaler (usa mediana/IQR).
    df = df.copy()
    scaler = RobustScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    X = df.drop(columns=["Class"]).values.astype("float32")
    y = df["Class"].values.astype("int32")

    # 60 % train / 20 % validación / 20 % test (estratificado).
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )

    print(
        f"Particiones -> train: {len(y_train):,} | "
        f"val: {len(y_val):,} | test: {len(y_test):,}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# --------------------------------------------------------------------------- #
# 3) Definición del modelo
# --------------------------------------------------------------------------- #
def build_model(input_dim: int, hidden=(32, 16), dropout=0.0, lr=1e-3):
    """Crea y compila un perceptrón multicapa (MLP) para clasificación binaria."""
    model = keras.Sequential(name="mlp_fraude")
    model.add(keras.Input(shape=(input_dim,)))
    for i, units in enumerate(hidden):
        model.add(layers.Dense(units, activation="relu", name=f"densa_{i + 1}"))
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f"dropout_{i + 1}"))
    model.add(layers.Dense(1, activation="sigmoid", name="salida"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# --------------------------------------------------------------------------- #
# 4) Pesos de clase y submuestreo (estrategias contra el desbalance)
# --------------------------------------------------------------------------- #
def class_weights(y):
    neg, pos = np.bincount(y)
    total = neg + pos
    # Cada clase pesa lo mismo en la función de pérdida.
    return {0: total / (2.0 * neg), 1: total / (2.0 * pos)}


def undersample(X, y, ratio=1.0, seed=SEED):
    """Submuestrea la clase mayoritaria. ratio = #negativos / #positivos."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg = min(len(neg_idx), int(len(pos_idx) * ratio))
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(keep)
    return X[keep], y[keep]


# --------------------------------------------------------------------------- #
# 5) Gráficas de loss y accuracy (entrenamiento y validación)
# --------------------------------------------------------------------------- #
def plot_history(history, name: str, outdir: str = RESULTS_DIR) -> str:
    os.makedirs(outdir, exist_ok=True)
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, h["loss"], "o-", label="Entrenamiento")
    ax1.plot(epochs, h["val_loss"], "s-", label="Validación")
    ax1.set_title(f"Loss — {name}")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (binary cross-entropy)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, h["accuracy"], "o-", label="Entrenamiento")
    ax2.plot(epochs, h["val_accuracy"], "s-", label="Validación")
    ax2.set_title(f"Accuracy — {name}")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"curvas_{name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Gráfica guardada -> {path}")
    return path


# --------------------------------------------------------------------------- #
# 6) Un experimento (un entrenamiento)
# --------------------------------------------------------------------------- #
def run_experiment(cfg, data, epochs, batch_size):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    name = cfg["name"]
    print("\n" + "=" * 70)
    print(f"EXPERIMENTO: {name}  —  {cfg['desc']}")
    print("=" * 70)

    keras.backend.clear_session()
    set_seeds()

    Xtr, ytr = X_train, y_train
    cw = None
    if cfg.get("undersample"):
        Xtr, ytr = undersample(X_train, y_train, ratio=cfg.get("ratio", 1.0))
        print(f"  Submuestreo -> nuevo train: {len(ytr):,} ({int(ytr.sum())} fraudes)")
    if cfg.get("class_weight"):
        cw = class_weights(y_train)
        print(f"  Pesos de clase -> {cw}")

    model = build_model(
        input_dim=X_train.shape[1],
        hidden=cfg.get("hidden", (32, 16)),
        dropout=cfg.get("dropout", 0.0),
        lr=cfg.get("lr", 1e-3),
    )

    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "arquitectura": cfg.get("hidden", (32, 16)),
                "dropout": cfg.get("dropout", 0.0),
                "learning_rate": cfg.get("lr", 1e-3),
                "epochs": epochs,
                "batch_size": batch_size,
                "class_weight": bool(cfg.get("class_weight")),
                "undersample": bool(cfg.get("undersample")),
            }
        )

        history = model.fit(
            Xtr,
            ytr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=cw,
            verbose=2,
        )

        # Curvas de loss/accuracy por época (train y val) -> MLflow.
        for metric, values in history.history.items():
            for step, v in enumerate(values):
                mlflow.log_metric(metric, float(v), step=step)

        png = plot_history(history, name)
        mlflow.log_artifact(png, artifact_path="curvas")

        # Evaluación final en el conjunto de TEST (datos nunca vistos).
        proba = model.predict(X_test, batch_size=4096, verbose=0).ravel()
        pred = (proba >= 0.5).astype(int)

        roc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        cm = confusion_matrix(y_test, pred)
        rep = classification_report(
            y_test, pred, target_names=["normal", "fraude"], output_dict=True
        )

        metrics = {
            "test_roc_auc": roc,
            "test_pr_auc": pr_auc,
            "test_precision_fraude": rep["fraude"]["precision"],
            "test_recall_fraude": rep["fraude"]["recall"],
            "test_f1_fraude": rep["fraude"]["f1-score"],
        }
        mlflow.log_metrics(metrics)

        print(f"  Matriz de confusión (test):\n{cm}")
        print(
            f"  ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f} | "
            f"Recall fraude: {metrics['test_recall_fraude']:.4f} | "
            f"Precision fraude: {metrics['test_precision_fraude']:.4f}"
        )

    return {"name": name, **metrics}


# --------------------------------------------------------------------------- #
# 7) Programa principal: VARIOS entrenamientos
# --------------------------------------------------------------------------- #
EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "MLP simple, SIN tratar el desbalance",
        "hidden": (32, 16),
        "dropout": 0.0,
    },
    {
        "name": "pesos_clase",
        "desc": "Mismo MLP + pesos de clase",
        "hidden": (32, 16),
        "dropout": 0.0,
        "class_weight": True,
    },
    {
        "name": "profundo_dropout",
        "desc": "Red más profunda + dropout + pesos de clase",
        "hidden": (64, 32, 16),
        "dropout": 0.3,
        "class_weight": True,
    },
    {
        "name": "submuestreo",
        "desc": "MLP entrenado sobre datos balanceados por submuestreo",
        "hidden": (32, 16),
        "dropout": 0.2,
        "undersample": True,
        "ratio": 1.0,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento RNN - fraude tarjetas")
    parser.add_argument("--data", default="archive/creditcard.csv", help="Ruta al CSV")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--experiment", default="fraude_tarjetas_rnn")
    args = parser.parse_args()

    set_seeds()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mlflow.set_experiment(args.experiment)

    data = load_and_prepare(args.data)

    resumen = [run_experiment(cfg, data, args.epochs, args.batch_size) for cfg in EXPERIMENTS]

    df = pd.DataFrame(resumen).set_index("name")
    df.to_csv(os.path.join(RESULTS_DIR, "resumen.csv"))
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO (conjunto de test)")
    print("=" * 70)
    print(df.round(4).to_string())
    print(f"\nGráficas y resumen en ./{RESULTS_DIR}/  |  Métricas en MLflow (./mlruns)")


if __name__ == "__main__":
    main()

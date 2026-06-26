# Detección de fraude en tarjetas de crédito con una Red Neuronal

Tarea de **Redes Neuronales Artificiales** — Dr. Jorge Velazquez-Castro.
Problema 2: **Detectar fraudes en movimientos de tarjetas bancarias**.

Se entrena un **Perceptrón Multicapa (TensorFlow/Keras)** sobre el dataset
[Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
probando **varias estrategias** para manejar el fuerte desbalance de clases, con
seguimiento de experimentos en **MLflow** y gráficas de **loss/accuracy** (entrenamiento y validación).

---

##  Contenido del repositorio

```
creditcard-fraud-nn/
├── notebooks/
│   └── fraude_tarjetas_colab.ipynb   # Cuaderno listo para Google Colab (recomendado)
├── src/
│   └── train.py                      # Mismo experimento como script de Python
├── report/
│   └── REPORTE.md                    # Reporte de la estrategia (entregable #1)
├── results/                          # Gráficas y resumen (se generan al entrenar)
├── requirements.txt
├── .gitignore
└── README.md
```

>  El dataset `creditcard.csv` (~144 MB) **no** se incluye en el repo (ver `.gitignore`).
> Descárgalo de Kaggle: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

---

## Opción A — Google Colab (recomendada)

1. Sube `notebooks/fraude_tarjetas_colab.ipynb` a [Google Colab](https://colab.research.google.com/)
   (o ábrelo desde GitHub: *File → Open notebook → GitHub*).
2. Menú **Entorno de ejecución → Ejecutar todo**.
3. Cuando se te pida, **sube el archivo `creditcard.csv`**.
4. Verás en pantalla las **curvas de loss y accuracy** de cada entrenamiento y la tabla
   comparativa final. Todo queda registrado en **MLflow** (`./mlruns`).
5. Para guardar las gráficas en este repo, ejecuta al final:
   ```python
   !zip -r resultados.zip results mlruns
   from google.colab import files; files.download('resultados.zip')
   ```
   y copia el contenido de `results/` a la carpeta `results/` del repositorio.

## Opción B — Local (PC con Python)

```bash
pip install -r requirements.txt
python src/train.py --data archive/creditcard.csv --epochs 40
mlflow ui          # abre http://127.0.0.1:5000 para ver las métricas
```

---

##  Experimentos (varios entrenamientos)

| Experimento | Técnica contra el desbalance |
|---|---|
| `baseline` | Ninguna (referencia) |
| `pesos_clase` | Pesos de clase en la pérdida |
| `profundo_dropout` | Red más profunda + dropout + pesos de clase |
| `submuestreo` | Datos balanceados por submuestreo |

Métrica principal: **recall / precision / PR-AUC del fraude** (el accuracy engaña por el
desbalance). Detalles en [`report/REPORTE.md`](report/REPORTE.md).

---

## Enlaces de entrega

> Todos los enlaces son **públicos**.

- **Repositorio (código):** <https://github.com/denissegc96-collab/creditcard-fraud-nn>
- **Reporte:** <https://github.com/denissegc96-collab/creditcard-fraud-nn/blob/main/report/REPORTE.md>
- **Gráficas (loss/accuracy):** <https://github.com/denissegc96-collab/creditcard-fraud-nn/tree/main/results>

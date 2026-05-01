# Automatización de un pipeline de ML con GitHub Actions 🚀

Este proyecto implementa un pipeline de Machine Learning reproducible para predecir el riesgo de ataques cardíacos, automatizado mediante CI/CD con GitHub Actions y MLflow.

## Estructura del Proyecto
- `data/raw/`: Dataset original.
- `data/processed/`: Dataset limpio y sin valores nulos.
- `src/`: Scripts modulares para preprocesamiento, entrenamiento y evaluación.
- `.github/workflows/`: Configuración del pipeline de CI/CD.

## Requisitos y Configuración Local
1. Crear un entorno virtual: `python -m venv venv`
2. Activar el entorno: `.\venv\Scripts\Activate.ps1` (Windows) o `source venv/bin/activate` (Linux/Mac)
3. Instalar dependencias: `pip install -r requirements.txt` o `make install`

## Ejecución del Pipeline
Puedes ejecutar el pipeline completo usando el Makefile:
- Limpieza y Entrenamiento: `make train` (o `python src/preprocess.py` seguido de `python src/train.py`).
- Pruebas: `make test` (o `pytest tests/`).

## Tracking con MLflow
El modelo (`RandomForestClassifier`) y sus métricas (Accuracy, F1-Score) quedan registrados localmente en la carpeta `mlruns/` y en `mlflow.db`. 
Para ver la interfaz gráfica de MLflow, ejecuta:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
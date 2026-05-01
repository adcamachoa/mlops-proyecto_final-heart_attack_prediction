import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature
import os
import warnings
from evaluate import calculate_metrics

warnings.filterwarnings("ignore")

def main():
    # 1. Cargar datos procesados
    data_path = "data/processed/heart_attack_cleaned.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró {data_path}. Ejecuta preprocess.py primero.")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=["num"])
    y = df["num"]

    # 2. División de datos (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Configuración de MLflow local
    # Usamos una base de datos SQLite local para asegurar que los registros se guarden correctamente
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Attack_Prediction")

    with mlflow.start_run():
        # Hiperparámetros
        n_estimators = 100
        max_depth = 5
        
        # Registrar parámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Entrenar el modelo
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Predicciones y Evaluación modular
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        # Registrar métricas
        mlflow.log_metrics(metrics)

        # Crear firma y ejemplo de entrada para MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[[0]]

        # Registrar el modelo
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="random_forest_model", 
            signature=signature, 
            input_example=input_example
        )
        
        print("✅ Entrenamiento completado.")
        print(f"📊 Métricas: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1_score']:.4f}")
        print("✅ Modelo y métricas registrados en MLflow exitosamente.")

if __name__ == "__main__":
    main()
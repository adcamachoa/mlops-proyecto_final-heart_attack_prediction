import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature

def main():
    # 1. Cargar configuración desde YAML
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    processed_data_path = config["data"]["processed_path"]
    print(f"Cargando datos preprocesados desde: {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    # 2. Separar variables (X) y la variable objetivo (y)
    # Solución al Data Leakage: Binarizar usando el umbral del config
    X = df.drop(columns=["target"])
    X = X.astype(float)
    y = (df["target"] > config["model"]["threshold"]).astype(int)
    
    # 3. División de datos (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["model"]["random_state"]
    )
    
    # 4. Configurar experimento en MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Attack_Prediction")
    
    with mlflow.start_run():
        # Obtener hiperparámetros del YAML
        n_estimators = config["model"]["n_estimators"]
        max_depth = config["model"]["max_depth"]
        random_state = config["model"]["random_state"]
        
        print(f"Entrenando RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
        
        # Entrenar modelo
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluar métricas
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"✅ Entrenamiento completado.")
        print(f"📊 Métricas: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")
        
        # --- 5. REGISTRO EN MLFLOW ---
        # Registrar hiperparámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("threshold", config["model"]["threshold"])
        
        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Generar firma y ejemplo de entrada (Buenas prácticas MLOps)
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(3)
        
        # Guardar el modelo como artefacto
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            signature=signature,
            input_example=input_example
        )
        print("✅ Modelo, firma y métricas registrados en MLflow exitosamente.")

if __name__ == "__main__":
    main()
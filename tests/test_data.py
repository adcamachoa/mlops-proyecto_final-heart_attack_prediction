import pandas as pd
import os

def test_no_nulls_in_processed_data():
    """Prueba que el dataset procesado no tenga valores nulos."""
    data_path = "data/processed/heart_attack_cleaned.csv"
    
    # Verificar que el archivo existe
    assert os.path.exists(data_path), "El archivo procesado no existe. Ejecuta preprocess.py primero."
    
    # Verificar que no hay nulos
    df = pd.read_csv(data_path)
    assert df.isnull().sum().sum() == 0, "Aún hay valores nulos en el dataset procesado."
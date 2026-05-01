import pandas as pd
import numpy as np
import os

def preprocess_data(input_path, output_dir):
    print(f"Cargando datos raw desde: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Limpiar posibles espacios en blanco en las columnas
    df.columns = df.columns.str.strip()
    
    # 2. Reemplazar '?' por nulos reales
    df.replace('?', np.nan, inplace=True)
    
    # 3. Forzar el tipo de dato numérico
    cols_to_numeric = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 4. Imputar nulos con la mediana
    df = df.fillna(df.median())
    
    # 5. Guardar el dataset limpio
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'heart_attack_cleaned.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Datos preprocesados y guardados en: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/raw/heart_attack_prediction.csv"
    OUTPUT_FOLDER = "data/processed/"
    preprocess_data(INPUT_FILE, OUTPUT_FOLDER)
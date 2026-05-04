import pandas as pd
import numpy as np
import yaml
import os

def preprocess_data(input_path, output_path):
    print(f"Cargando datos raw desde: {input_path}")
    df = pd.read_csv(input_path)
    
    print("Limpiando valores nulos y estandarizando tipos de datos...")
    # Reemplazar '?' ocultos por valores nulos reales (NaN)
    df.replace('?', np.nan, inplace=True)
    
    # Forzar todas las columnas a formato numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Imputar valores faltantes usando la mediana
    df = df.fillna(df.median())
    
    # Extraemos solo la ruta de la carpeta (ej. "data/processed")
    directorio = os.path.dirname(output_path)
    if directorio: # Solo crea la carpeta si la ruta no está vacía
        os.makedirs(directorio, exist_ok=True)
    
    # Guardar el archivo limpio
    df.to_csv(output_path, index=False)
    print(f"✅ Datos preprocesados y guardados en: {output_path}")

if __name__ == "__main__":
    # Cargar las rutas desde config.yaml
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    INPUT_FILE = config["data"]["raw_path"]
    OUTPUT_FILE = config["data"]["processed_path"]
    
    preprocess_data(INPUT_FILE, OUTPUT_FILE)
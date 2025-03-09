import pandas as pd

# Leer el archivo CSV
df = pd.read_csv("datos.csv", quoting=1)  # quoting=1 para manejar comillas en texto

# Verificar estructura
print(df["Departamento"].head())
#print(df.head())
#print("\nDistribución de departamentos:")
#print(df["Departamento"].value_counts())
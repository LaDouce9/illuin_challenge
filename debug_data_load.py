import pandas as pd
import json
from pathlib import Path

# Chemin vers les données
data_dir = Path('data/raw/code_classification_dataset')

print(f"Checking directory: {data_dir.absolute()}")
if not data_dir.exists():
    print("Directory does not exist!")
    exit(1)

# Chargement des fichiers JSON
data = []
files = list(data_dir.glob('*.json'))
print(f"Found {len(files)} JSON files.")

if len(files) == 0:
    print("No JSON files found!")
    exit(1)

# Load first 5 for testing
for file_path in files[:5]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Création du DataFrame
df = pd.DataFrame(data)

# Affichage des premières lignes
print(df.head())
print(df.info())

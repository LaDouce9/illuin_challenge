"""
Script to execute and validate the EDA notebook
"""
import sys
sys.path.append('/app')

import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.utils.eda_helpers import (
    load_dataset, extract_tags_list, get_tag_statistics,
    get_priority_tag_coverage, PRIORITY_TAGS
)

print("=" * 80)
print("VALIDATION - CHARGEMENT DES DONNÉES")
print("=" * 80)

# Test 1: Chargement du dataset
try:
    DATA_DIR = '/app/data/raw/code_classification_dataset'
    df = load_dataset(DATA_DIR)
    print(f"✅ Dataset chargé: {df.shape}")
    print(f"   Colonnes: {list(df.columns)[:5]}... ({len(df.columns)} total)")
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)

# Test 2: Parsing des tags
try:
    df = extract_tags_list(df)
    print(f"✅ Tags parsés")
    print(f"   Exemple de tags: {df['tags'].iloc[0]}")
except Exception as e:
    print(f"❌ Erreur lors du parsing des tags: {e}")
    sys.exit(1)

# Test 3: Statistiques des tags
try:
    tag_stats = get_tag_statistics(df)
    print(f"✅ Statistiques calculées")
    print(f"   Nombre de tags uniques: {len(tag_stats)}")
    print(f"\n   Top 10 tags:")
    print(tag_stats.head(10)[['tag', 'count', 'is_priority']].to_string(index=False))
except Exception as e:
    print(f"❌ Erreur lors du calcul des statistiques: {e}")
    sys.exit(1)

# Test 4: Tags prioritaires
try:
    priority_stats = tag_stats[tag_stats['is_priority']].copy()
    print(f"\n✅ Tags prioritaires identifiés: {len(priority_stats)}")
    print(f"\n   Distribution des tags prioritaires:")
    print(priority_stats[['tag', 'count', 'frequency']].to_string(index=False))
except Exception as e:
    print(f"❌ Erreur avec les tags prioritaires: {e}")
    sys.exit(1)

# Test 5: Couverture
try:
    coverage = get_priority_tag_coverage(df)
    print(f"\n✅ Couverture calculée:")
    for key, value in coverage.items():
        print(f"   {key}: {value}")
except Exception as e:
    print(f"❌ Erreur lors du calcul de la couverture: {e}")
    sys.exit(1)

# Test 6: Valeurs manquantes
print(f"\n" + "=" * 80)
print("VALIDATION - QUALITÉ DES DONNÉES")
print("=" * 80)
missing = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
}).sort_values('Missing_Count', ascending=False)

missing_cols = missing[missing['Missing_Count'] > 0]
if len(missing_cols) > 0:
    print(f"⚠️  Valeurs manquantes détectées:")
    print(missing_cols.to_string(index=False))
else:
    print(f"✅ Aucune valeur manquante")

# Test 7: Distribution des langages
print(f"\n" + "=" * 80)
print("VALIDATION - DISTRIBUTION DES LANGAGES")
print("=" * 80)
lang_counts = df['lang'].value_counts()
print(lang_counts)

print(f"\n" + "=" * 80)
print("✅ VALIDATION COMPLÈTE - TOUS LES TESTS PASSÉS")
print("=" * 80)

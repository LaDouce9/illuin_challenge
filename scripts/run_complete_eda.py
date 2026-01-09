"""
Complete EDA Execution Script
Executes all analyses from the notebook and generates outputs
"""
import sys
sys.path.append('/app')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.utils.eda_helpers import (
    load_dataset, extract_tags_list, get_tag_statistics,
    get_priority_tag_coverage, analyze_tag_cooccurrence,
    get_tag_examples, compute_text_statistics, analyze_code_complexity,
    plot_tag_distribution, plot_cooccurrence_heatmap, PRIORITY_TAGS
)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("DÉMARRAGE DE L'ANALYSE EDA COMPLÈTE")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/10] Chargement des données...")
DATA_DIR = '/app/data/raw/code_classification_dataset'
df = load_dataset(DATA_DIR)
print(f"✅ Dataset chargé: {df.shape[0]} échantillons, {df.shape[1]} colonnes")

# ============================================================================
# 2. PARSING DES TAGS
# ============================================================================
print("\n[2/10] Parsing des tags...")
df = extract_tags_list(df)
tag_stats = get_tag_statistics(df)
print(f"✅ {len(tag_stats)} tags uniques identifiés")

# ============================================================================
# 3. ANALYSE DES VALEURS MANQUANTES
# ============================================================================
print("\n[3/10] Analyse des valeurs manquantes...")
missing = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
}).sort_values('Missing_Count', ascending=False)

missing_cols = missing[missing['Missing_Count'] > 0]
if len(missing_cols) > 0:
    print(f"⚠️  {len(missing_cols)} colonnes avec valeurs manquantes:")
    for _, row in missing_cols.head(5).iterrows():
        print(f"   - {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")
else:
    print("✅ Aucune valeur manquante")

# ============================================================================
# 4. ANALYSE DES TAGS PRIORITAIRES
# ============================================================================
print("\n[4/10] Analyse des tags prioritaires...")
priority_stats = tag_stats[tag_stats['is_priority']].copy()
coverage = get_priority_tag_coverage(df)

print(f"✅ Tags prioritaires trouvés: {len(priority_stats)}/8")
print(f"   Couverture: {coverage['samples_with_priority_tag']}/{coverage['total_samples']} " +
      f"({coverage['coverage_percentage']:.1f}%)")

print("\n   Distribution:")
for _, row in priority_stats.iterrows():
    print(f"   - {row['tag']:20s}: {row['count']:4d} ({row['frequency']*100:5.1f}%)")

# ============================================================================
# 5. CO-OCCURRENCE DES TAGS
# ============================================================================
print("\n[5/10] Analyse de co-occurrence...")
cooccurrence = analyze_tag_cooccurrence(df, PRIORITY_TAGS)

# Paires les plus fréquentes
from itertools import combinations
from collections import Counter

tag_pairs = []
for tags in df['tags']:
    priority_tags_in_sample = [t for t in tags if t in PRIORITY_TAGS]
    if len(priority_tags_in_sample) >= 2:
        tag_pairs.extend(list(combinations(sorted(priority_tags_in_sample), 2)))

pair_counts = Counter(tag_pairs)
print(f"✅ Top 5 paires de tags:")
for pair, count in pair_counts.most_common(5):
    print(f"   - {pair[0]:15s} + {pair[1]:15s}: {count:3d} occurrences")

# ============================================================================
# 6. ANALYSE PAR TAG
# ============================================================================
print("\n[6/10] Analyse par tag prioritaire...")
tag_analysis = []

for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    tag_df = df[mask]
    
    analysis = {
        'tag': tag,
        'count': len(tag_df),
        'avg_difficulty': tag_df['difficulty'].mean() if 'difficulty' in tag_df.columns else None,
        'avg_code_length': tag_df['source_code'].str.len().mean(),
        'avg_desc_length': tag_df['prob_desc_description'].str.len().mean(),
    }
    tag_analysis.append(analysis)

tag_analysis_df = pd.DataFrame(tag_analysis)
print("✅ Caractéristiques moyennes calculées")

# ============================================================================
# 7. ANALYSE NLP DES DESCRIPTIONS
# ============================================================================
print("\n[7/10] Analyse NLP des descriptions...")
import re

def clean_text(text):
    text = re.sub(r'\$\$\$.*?\$\$\$', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_description'] = df['prob_desc_description'].apply(clean_text)

# Mots-clés par tag
from sklearn.feature_extraction.text import CountVectorizer

keywords_by_tag = {}
for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    tag_descriptions = df[mask]['clean_description'].tolist()
    
    if len(tag_descriptions) > 0:
        vectorizer = CountVectorizer(max_features=10, stop_words='english')
        try:
            X = vectorizer.fit_transform(tag_descriptions)
            word_counts = X.sum(axis=0).A1
            words = vectorizer.get_feature_names_out()
            keywords_by_tag[tag] = list(zip(words, word_counts))
        except:
            keywords_by_tag[tag] = []

print("✅ Mots-clés extraits pour chaque tag")

# ============================================================================
# 8. ANALYSE DU CODE SOURCE
# ============================================================================
print("\n[8/10] Analyse du code source...")
code_stats = compute_text_statistics(df['source_code'])
df = pd.concat([df, code_stats], axis=1)

print(f"✅ Statistiques de longueur calculées")
print(f"   Longueur moyenne: {df['length_chars'].mean():.0f} caractères, " +
      f"{df['length_lines'].mean():.0f} lignes")

# Complexité sur échantillon
sample_df = df.sample(min(100, len(df)), random_state=42)
complexity_results = sample_df['source_code'].apply(analyze_code_complexity)
complexity_df = pd.DataFrame(complexity_results.tolist())

print(f"   Complexité moyenne (échantillon):")
print(f"   - Fonctions: {complexity_df['num_functions'].mean():.1f}")
print(f"   - Boucles: {complexity_df['num_for_loops'].mean() + complexity_df['num_while_loops'].mean():.1f}")
print(f"   - Conditions: {complexity_df['num_if_statements'].mean():.1f}")

# ============================================================================
# 9. DISTRIBUTION DES LANGAGES
# ============================================================================
print("\n[9/10] Distribution des langages...")
lang_counts = df['lang'].value_counts()
print("✅ Distribution:")
for lang, count in lang_counts.items():
    print(f"   - {lang:15s}: {count:4d} ({count/len(df)*100:5.1f}%)")

# ============================================================================
# 10. GÉNÉRATION DES VISUALISATIONS
# ============================================================================
print("\n[10/10] Génération des visualisations...")

# Viz 1: Distribution des tags prioritaires
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(priority_stats['tag'], priority_stats['count'], color='#2ecc71', alpha=0.8)
ax.set_xlabel('Tag', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution des 8 Tags Prioritaires', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('/app/docs/priority_tags_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Viz 2: Heatmap co-occurrence
fig = plot_cooccurrence_heatmap(cooccurrence)
plt.savefig('/app/docs/cooccurrence_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Viz 3: Caractéristiques par tag
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if tag_analysis_df['avg_difficulty'].notna().any():
    axes[0, 0].bar(tag_analysis_df['tag'], tag_analysis_df['avg_difficulty'], color='#9b59b6', alpha=0.7)
    axes[0, 0].set_ylabel('Difficulté Moyenne', fontsize=11)
    axes[0, 0].set_title('Difficulté Moyenne par Tag', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(tag_analysis_df['tag'], tag_analysis_df['avg_code_length'], color='#e67e22', alpha=0.7)
axes[0, 1].set_ylabel('Longueur Moyenne (caractères)', fontsize=11)
axes[0, 1].set_title('Longueur Moyenne du Code par Tag', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].bar(tag_analysis_df['tag'], tag_analysis_df['avg_desc_length'], color='#1abc9c', alpha=0.7)
axes[1, 0].set_ylabel('Longueur Moyenne (caractères)', fontsize=11)
axes[1, 0].set_title('Longueur Moyenne de la Description par Tag', fontsize=12, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].bar(tag_analysis_df['tag'], tag_analysis_df['count'], color='#34495e', alpha=0.7)
axes[1, 1].set_ylabel("Nombre d'échantillons", fontsize=11)
axes[1, 1].set_title("Nombre d'Échantillons par Tag", fontsize=12, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/app/docs/tag_characteristics.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ 3 visualisations sauvegardées dans /app/docs/")

# ============================================================================
# 11. SAUVEGARDE DU DATASET ENRICHI
# ============================================================================
print("\n[11/11] Sauvegarde du dataset enrichi...")
output_path = '/app/data/processed/dataset_with_eda_features.parquet'
df.to_parquet(output_path, index=False)
print(f"✅ Dataset sauvegardé: {output_path}")

print("\n" + "=" * 80)
print("✅ ANALYSE EDA COMPLÈTE TERMINÉE AVEC SUCCÈS")
print("=" * 80)
print(f"\nFichiers générés:")
print(f"  - /app/docs/priority_tags_distribution.png")
print(f"  - /app/docs/cooccurrence_heatmap.png")
print(f"  - /app/docs/tag_characteristics.png")
print(f"  - /app/data/processed/dataset_with_eda_features.parquet")

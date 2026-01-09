"""
Advanced EDA - Data Validity Checks & Critical Controls
Based on ChatGPT review recommendations
"""
import sys
sys.path.append('/app')

import pandas as pd
import numpy as np
import re
from collections import Counter
import json

print("=" * 80)
print("CONTRÔLES DE VALIDITÉ CRITIQUES")
print("=" * 80)

# Load dataset
df = pd.read_parquet('/app/data/processed/dataset_with_eda_features.parquet')
PRIORITY_TAGS = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']

# ============================================================================
# 1. UNICITÉ ET DÉTECTION DE DOUBLONS
# ============================================================================
print("\n[1/8] Contrôle d'unicité...")

# 1.1 Unicité par src_uid
src_uid_counts = df['src_uid'].value_counts()
duplicates_src = src_uid_counts[src_uid_counts > 1]
print(f"✓ src_uid uniques: {df['src_uid'].nunique()}/{len(df)}")
if len(duplicates_src) > 0:
    print(f"⚠️  {len(duplicates_src)} src_uid dupliqués:")
    print(duplicates_src.head())
else:
    print("✅ Tous les src_uid sont uniques")

# 1.2 Unicité par code_uid
code_uid_counts = df['code_uid'].value_counts()
duplicates_code = code_uid_counts[code_uid_counts > 1]
print(f"\n✓ code_uid uniques: {df['code_uid'].nunique()}/{len(df)}")
if len(duplicates_code) > 0:
    print(f"⚠️  {len(duplicates_code)} code_uid dupliqués")
else:
    print("✅ Tous les code_uid sont uniques")

# 1.3 Near-duplicates de description (hashing simple)
from hashlib import md5

def normalize_text(text):
    """Normalisation agressive pour détecter near-duplicates"""
    text = re.sub(r'\$\$\$.*?\$\$\$', '', text)  # Remove LaTeX
    text = re.sub(r'[^a-z\s]', '', text.lower())  # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['desc_hash'] = df['prob_desc_description'].apply(lambda x: md5(normalize_text(x).encode()).hexdigest())
desc_hash_counts = df['desc_hash'].value_counts()
duplicates_desc = desc_hash_counts[desc_hash_counts > 1]

print(f"\n✓ Descriptions normalisées uniques: {df['desc_hash'].nunique()}/{len(df)}")
if len(duplicates_desc) > 0:
    print(f"⚠️  {len(duplicates_desc)} groupes de descriptions similaires ({duplicates_desc.sum()} échantillons)")
    print("  Top 3 groupes:")
    for hash_val, count in duplicates_desc.head(3).items():
        print(f"    - {count} échantillons avec hash {hash_val[:8]}...")
else:
    print("✅ Toutes les descriptions sont uniques (après normalisation)")

# 1.4 Near-duplicates de code
df['code_hash'] = df['source_code'].apply(lambda x: md5(normalize_text(x).encode()).hexdigest())
code_hash_counts = df['code_hash'].value_counts()
duplicates_code_hash = code_hash_counts[code_hash_counts > 1]

print(f"\n✓ Codes normalisés uniques: {df['code_hash'].nunique()}/{len(df)}")
if len(duplicates_code_hash) > 0:
    print(f"⚠️  {len(duplicates_code_hash)} groupes de codes similaires ({duplicates_code_hash.sum()} échantillons)")
else:
    print("✅ Tous les codes sont uniques (après normalisation)")

# ============================================================================
# 2. NORMALISATION DES TAGS
# ============================================================================
print("\n[2/8] Audit de normalisation des tags...")

# Extraire tous les tags bruts
all_tags_raw = [tag for tags in df['tags'] for tag in tags]
all_tags_normalized = [tag.strip().lower() for tag in all_tags_raw]

# Comparer
unique_raw = set(all_tags_raw)
unique_normalized = set(all_tags_normalized)

print(f"✓ Tags bruts uniques: {len(unique_raw)}")
print(f"✓ Tags normalisés uniques: {len(unique_normalized)}")

if len(unique_raw) != len(unique_normalized):
    print(f"⚠️  Différence détectée: {len(unique_raw) - len(unique_normalized)} tags ont des variantes")
    # Trouver les variantes
    tag_variants = {}
    for tag_raw in unique_raw:
        tag_norm = tag_raw.strip().lower()
        if tag_norm not in tag_variants:
            tag_variants[tag_norm] = []
        tag_variants[tag_norm].append(tag_raw)
    
    variants_found = {k: v for k, v in tag_variants.items() if len(v) > 1}
    if variants_found:
        print(f"  Variantes trouvées:")
        for norm, variants in list(variants_found.items())[:5]:
            print(f"    - '{norm}': {variants}")
else:
    print("✅ Tous les tags sont déjà normalisés")

# Tags prioritaires: vérifier les variantes
print(f"\n✓ Vérification des tags prioritaires:")
for tag in PRIORITY_TAGS:
    count = sum(1 for tags in df['tags'] for t in tags if t.lower().strip() == tag.lower())
    print(f"  - '{tag}': {count} occurrences")

# ============================================================================
# 3. LANGUE DES TEXTES
# ============================================================================
print("\n[3/8] Détection de langue...")

# Analyse simple: ratio ASCII vs non-ASCII
def analyze_language(text):
    if not text or len(text) == 0:
        return {'ascii_ratio': 0, 'has_non_ascii': False}
    
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / len(text)
    has_non_ascii = ascii_ratio < 0.95
    
    return {'ascii_ratio': ascii_ratio, 'has_non_ascii': has_non_ascii}

df['desc_lang_info'] = df['prob_desc_description'].apply(analyze_language)
df['desc_ascii_ratio'] = df['desc_lang_info'].apply(lambda x: x['ascii_ratio'])
df['desc_has_non_ascii'] = df['desc_lang_info'].apply(lambda x: x['has_non_ascii'])

non_ascii_count = df['desc_has_non_ascii'].sum()
print(f"✓ Descriptions avec caractères non-ASCII: {non_ascii_count}/{len(df)} ({non_ascii_count/len(df)*100:.1f}%)")

if non_ascii_count > 0:
    print(f"  Ratio ASCII moyen (descriptions avec non-ASCII): {df[df['desc_has_non_ascii']]['desc_ascii_ratio'].mean():.3f}")
    # Exemples
    print(f"  Exemples (3 premiers):")
    for idx in df[df['desc_has_non_ascii']].head(3).index:
        desc = df.loc[idx, 'prob_desc_description'][:100]
        print(f"    - {desc}...")

# ============================================================================
# 4. LATEX: FEATURES QUANTITATIVES
# ============================================================================
print("\n[4/8] Analyse LaTeX...")

def extract_latex_features(text):
    """Extract LaTeX-related features"""
    latex_blocks = re.findall(r'\$\$\$.*?\$\$\$', text)
    latex_symbols = re.findall(r'\\(frac|sum|prod|int|mod|gcd|lcm|sqrt|log|sin|cos|tan|prime)', text)
    
    return {
        'nb_latex_blocks': len(latex_blocks),
        'nb_latex_symbols': len(latex_symbols),
        'total_latex_chars': sum(len(b) for b in latex_blocks),
        'latex_density': sum(len(b) for b in latex_blocks) / len(text) if len(text) > 0 else 0
    }

df['latex_features'] = df['prob_desc_description'].apply(extract_latex_features)
df['nb_latex_blocks'] = df['latex_features'].apply(lambda x: x['nb_latex_blocks'])
df['nb_latex_symbols'] = df['latex_features'].apply(lambda x: x['nb_latex_symbols'])
df['latex_density'] = df['latex_features'].apply(lambda x: x['latex_density'])

print(f"✓ Échantillons avec LaTeX: {(df['nb_latex_blocks'] > 0).sum()}/{len(df)} ({(df['nb_latex_blocks'] > 0).sum()/len(df)*100:.1f}%)")
print(f"  Moyenne blocs LaTeX: {df['nb_latex_blocks'].mean():.2f}")
print(f"  Moyenne symboles LaTeX: {df['nb_latex_symbols'].mean():.2f}")
print(f"  Densité LaTeX moyenne: {df['latex_density'].mean():.4f}")

# Par tag prioritaire
print(f"\n  Densité LaTeX par tag prioritaire:")
for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    if mask.sum() > 0:
        avg_density = df[mask]['latex_density'].mean()
        print(f"    - {tag:20s}: {avg_density:.4f}")

# ============================================================================
# 5. EXEC_OUTCOME: ANALYSE DE LEAKAGE
# ============================================================================
print("\n[5/8] Analyse exec_outcome (risque de leakage)...")

exec_counts = df['exec_outcome'].value_counts()
print(f"✓ Distribution exec_outcome:")
for outcome, count in exec_counts.items():
    print(f"  - {outcome}: {count} ({count/len(df)*100:.1f}%)")

# Distribution par tag
print(f"\n  Distribution exec_outcome par tag prioritaire:")
for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    tag_df = df[mask]
    if len(tag_df) > 0:
        passed_ratio = (tag_df['exec_outcome'] == 'PASSED').sum() / len(tag_df)
        print(f"    - {tag:20s}: {passed_ratio*100:.1f}% PASSED")

# ============================================================================
# 6. MISSINGNESS NOT AT RANDOM
# ============================================================================
print("\n[6/8] Analyse missingness (prob_desc_notes)...")

df['notes_is_missing'] = df['prob_desc_notes'].isnull()
missing_ratio = df['notes_is_missing'].sum() / len(df)
print(f"✓ Notes manquantes: {df['notes_is_missing'].sum()}/{len(df)} ({missing_ratio*100:.1f}%)")

# Par tag
print(f"\n  Taux de notes manquantes par tag prioritaire:")
for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    tag_df = df[mask]
    if len(tag_df) > 0:
        missing_tag_ratio = tag_df['notes_is_missing'].sum() / len(tag_df)
        print(f"    - {tag:20s}: {missing_tag_ratio*100:.1f}%")

# Par difficulté
if 'difficulty' in df.columns:
    df['difficulty_bin'] = pd.cut(df['difficulty'], bins=[0, 1200, 1600, 2000, 3500], labels=['Easy', 'Medium', 'Hard', 'Very Hard'])
    print(f"\n  Taux de notes manquantes par difficulté:")
    for bin_name in ['Easy', 'Medium', 'Hard', 'Very Hard']:
        bin_df = df[df['difficulty_bin'] == bin_name]
        if len(bin_df) > 0:
            missing_bin_ratio = bin_df['notes_is_missing'].sum() / len(bin_df)
            print(f"    - {bin_name:15s}: {missing_bin_ratio*100:.1f}%")

# ============================================================================
# 7. LABEL DENSITY ET COMPLEXITÉ
# ============================================================================
print("\n[7/8] Analyse multi-label avancée...")

# Label density
df['num_tags'] = df['tags'].apply(len)
df['label_density'] = df['num_tags'] / 37  # 37 tags uniques

print(f"✓ Label density moyenne: {df['label_density'].mean():.3f}")
print(f"  Min: {df['label_density'].min():.3f}, Max: {df['label_density'].max():.3f}")

# Top combinaisons
tag_combinations = df['tags'].apply(lambda x: tuple(sorted(x)))
combo_counts = tag_combinations.value_counts()

print(f"\n✓ Combinaisons uniques: {len(combo_counts)}")
print(f"  Top 10 combinaisons:")
for combo, count in combo_counts.head(10).items():
    print(f"    - {combo}: {count}")

# ============================================================================
# 8. SAUVEGARDE DES RÉSULTATS
# ============================================================================
print("\n[8/8] Sauvegarde des résultats...")

validation_results = {
    "uniqueness": {
        "src_uid_unique": df['src_uid'].nunique() == len(df),
        "code_uid_unique": df['code_uid'].nunique() == len(df),
        "desc_duplicates": len(duplicates_desc),
        "code_duplicates": len(duplicates_code_hash)
    },
    "tags_normalization": {
        "raw_unique": len(unique_raw),
        "normalized_unique": len(unique_normalized),
        "has_variants": len(unique_raw) != len(unique_normalized)
    },
    "language": {
        "non_ascii_count": int(non_ascii_count),
        "non_ascii_percentage": float(non_ascii_count/len(df)*100)
    },
    "latex": {
        "samples_with_latex": int((df['nb_latex_blocks'] > 0).sum()),
        "avg_latex_density": float(df['latex_density'].mean())
    },
    "exec_outcome": {
        "passed_percentage": float((df['exec_outcome'] == 'PASSED').sum() / len(df) * 100)
    },
    "missingness": {
        "notes_missing_percentage": float(missing_ratio * 100)
    },
    "label_complexity": {
        "avg_label_density": float(df['label_density'].mean()),
        "unique_combinations": len(combo_counts)
    }
}

with open('/app/docs/validation_results.json', 'w') as f:
    json.dump(validation_results, f, indent=2)

# Sauvegarder le dataset enrichi avec nouvelles features
df.to_parquet('/app/data/processed/dataset_validated.parquet', index=False)

print(f"\n✅ Validation complète terminée")
print(f"  - Résultats: /app/docs/validation_results.json")
print(f"  - Dataset: /app/data/processed/dataset_validated.parquet")
print(f"  - Nouvelles features: latex_density, notes_is_missing, label_density")

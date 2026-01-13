"""
Module pour les analyses EDA avancées.

Ce module contient des fonctions pour :
- Détection de near-duplicates (textes très similaires)
- Analyse de co-occurrence avancée (Lift, PMI)
- Distribution détaillée des tags

Auteur : EDA Team
Date : Janvier 2026
"""

import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from hashlib import md5
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Dict, Tuple, Optional


# ============================================================================
# SECTION 1 : NEAR-DUPLICATES
# ============================================================================

def normalize_text_for_duplicates(text: str) -> str:
    """
    Normalisation agressive pour détecter les near-duplicates.
    
    Cette fonction :
    - Supprime tout le LaTeX
    - Garde uniquement les lettres minuscules
    - Normalise les espaces
    
    Args:
        text (str): Texte à normaliser
        
    Returns:
        str: Texte normalisé
    """
    if pd.isna(text):
        return ""
    
    # Supprimer le LaTeX
    text = re.sub(r'\$\$\$.*?\$\$\$', '', text)
    text = re.sub(r'\$.*?\$', '', text)
    
    # Garder uniquement les lettres minuscules
    text = re.sub(r'[^a-z\s]', '', text.lower())
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def detect_near_duplicates(df: pd.DataFrame, 
                            column: str,
                            min_group_size: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Détecte les near-duplicates dans une colonne via hashing MD5.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Colonne à analyser
        min_group_size (int): Taille minimale d'un groupe de duplicates (défaut: 2)
        
    Returns:
        tuple: (duplicate_groups_df, df_with_hash)
            - duplicate_groups_df: DataFrame des groupes de duplicates
            - df_with_hash: DataFrame original avec colonne hash ajoutée
            
    Examples:
        >>> dup_groups, df_hash = detect_near_duplicates(df, 'prob_desc_description')
        >>> print(f"Trouvé {len(dup_groups)} groupes de duplicates")
    """
    df = df.copy()
    hash_column = f"{column}_hash"
    
    print(f"🔍 Détection de near-duplicates sur '{column}'...")
    
    # Normaliser et hasher
    df[hash_column] = df[column].apply(
        lambda x: md5(normalize_text_for_duplicates(x).encode()).hexdigest()
    )
    
    # Compter les occurrences de chaque hash
    hash_counts = df[hash_column].value_counts()
    duplicates_hashes = hash_counts[hash_counts >= min_group_size]
    
    # Statistiques
    n_unique = df[hash_column].nunique()
    n_total = len(df)
    n_dup_groups = len(duplicates_hashes)
    n_dup_samples = duplicates_hashes.sum()
    
    print(f"\n📊 Résultats:")
    print(f"   Textes normalisés uniques: {n_unique}/{n_total} ({n_unique/n_total*100:.1f}%)")
    
    if n_dup_groups > 0:
        print(f"   ⚠️  {n_dup_groups} groupes de near-duplicates détectés")
        print(f"   📝 {n_dup_samples} échantillons concernés ({n_dup_samples/n_total*100:.1f}%)")
        
        # Créer un DataFrame des groupes de duplicates
        duplicate_groups = []
        for hash_val, count in duplicates_hashes.items():
            group_df = df[df[hash_column] == hash_val]
            duplicate_groups.append({
                'hash': hash_val,
                'count': count,
                'indices': list(group_df.index),
                'tags_variety': len(set([tuple(sorted(tags)) for tags in group_df['tags']])) if 'tags' in df.columns else None
            })
        
        dup_groups_df = pd.DataFrame(duplicate_groups).sort_values('count', ascending=False)
        return dup_groups_df, df
    else:
        print("   ✅ Aucun near-duplicate détecté")
        return pd.DataFrame(), df


def display_duplicate_examples(df: pd.DataFrame,
                                 dup_groups_df: pd.DataFrame,
                                 column: str,
                                 n_groups: int = 3,
                                 max_text_length: int = 300) -> None:
    """
    Affiche des exemples de groupes de near-duplicates.
    
    Args:
        df (pd.DataFrame): DataFrame original
        dup_groups_df (pd.DataFrame): DataFrame des groupes (retourné par detect_near_duplicates)
        column (str): Nom de la colonne analysée
        n_groups (int): Nombre de groupes à afficher (défaut: 3)
        max_text_length (int): Longueur max du texte affiché (défaut: 300)
    """
    if len(dup_groups_df) == 0:
        print("Aucun groupe de duplicates à afficher")
        return
    
    print(f"\n{'='*100}")
    print(f"EXEMPLES DE NEAR-DUPLICATES (Top {n_groups} groupes)")
    print(f"{'='*100}")
    
    for i, row in dup_groups_df.head(n_groups).iterrows():
        hash_val = row['hash']
        count = row['count']
        indices = row['indices']
        
        print(f"\n{'─'*100}")
        print(f"Groupe {i+1} : {count} échantillons similaires")
        print(f"{'─'*100}")
        
        hash_column = f"{column}_hash"
        group_samples = df[df[hash_column] == hash_val]
        
        for idx, sample_row in group_samples.iterrows():
            tags = sample_row['tags'] if 'tags' in df.columns else []
            text = str(sample_row[column])
            
            # Tronquer le texte
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            print(f"\n   📄 Échantillon {idx}:")
            print(f"      Tags: {tags}")
            print(f"      Texte: {text}")
        
        print()


# ============================================================================
# SECTION 2 : CO-OCCURRENCE AVANCÉE (LIFT & PMI)
# ============================================================================

def compute_lift_pmi(df: pd.DataFrame,
                     tags_column: str = 'tags',
                     priority_tags: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calcule le Lift et le PMI pour toutes les paires de tags.
    
    **Lift** : P(A,B) / (P(A) × P(B))
    - Lift > 1 : Les tags apparaissent ensemble plus que par hasard
    - Lift = 1 : Indépendance
    - Lift < 1 : Les tags apparaissent ensemble moins que par hasard
    
    **PMI** (Pointwise Mutual Information) : log(Lift)
    - Mesure similaire mais sur échelle logarithmique
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne de tags
        tags_column (str): Nom de la colonne contenant les listes de tags
        priority_tags (List[str], optional): Filtrer sur ces tags uniquement
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes ['tag_a', 'tag_b', 'count', 'lift', 'pmi']
        
    Examples:
        >>> lift_pmi_df = compute_lift_pmi(df, tags_column='tags', priority_tags=PRIORITY_TAGS)
        >>> top_lift = lift_pmi_df.nlargest(10, 'lift')
    """
    n_total = len(df)
    
    # Calculer les probabilités individuelles P(tag)
    tag_counts = Counter()
    for tags in df[tags_column]:
        if isinstance(tags, list):
            tag_counts.update(tags)
    
    tag_probs = {tag: count / n_total for tag, count in tag_counts.items()}
    
    # Filtrer sur les tags prioritaires si spécifié
    if priority_tags:
        tags_to_analyze = [t for t in priority_tags if t in tag_probs]
    else:
        tags_to_analyze = list(tag_probs.keys())
    
    # Calculer les co-occurrences et Lift/PMI
    results = []
    
    for tag_a, tag_b in combinations(sorted(tags_to_analyze), 2):
        # Compter les co-occurrences
        cooccur_count = df[tags_column].apply(
            lambda tags: isinstance(tags, list) and tag_a in tags and tag_b in tags
        ).sum()
        
        # P(A,B)
        p_both = cooccur_count / n_total
        
        # Lift = P(A,B) / (P(A) * P(B))
        p_a = tag_probs[tag_a]
        p_b = tag_probs[tag_b]
        
        if p_a * p_b > 0:
            lift = p_both / (p_a * p_b)
        else:
            lift = 0
        
        # PMI = log(Lift)
        pmi = np.log(lift) if lift > 0 else 0
        
        results.append({
            'tag_a': tag_a,
            'tag_b': tag_b,
            'count': cooccur_count,
            'lift': lift,
            'pmi': pmi
        })
    
    results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
    
    return results_df


def display_lift_pmi_analysis(lift_pmi_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Affiche une analyse détaillée des résultats Lift/PMI.
    
    Args:
        lift_pmi_df (pd.DataFrame): DataFrame retourné par compute_lift_pmi()
        top_n (int): Nombre de paires à afficher (défaut: 10)
    """
    print("=" * 100)
    print("CO-OCCURRENCE AVANCÉE: LIFT & PMI")
    print("=" * 100)
    
    print(f"\n✓ Top {top_n} paires par Lift (association forte):")
    print(f"  {'Tag A':18s} + {'Tag B':18s}   Lift    PMI   Count")
    print(f"  {'-'*18}   {'-'*18}   {'─'*4}    {'─'*4}  {'─'*5}")
    
    for _, row in lift_pmi_df.head(top_n).iterrows():
        print(f"  {row['tag_a']:18s} + {row['tag_b']:18s}   "
              f"{row['lift']:5.2f}  {row['pmi']:6.2f}  {row['count']:5d}")
    
    # Insight
    if len(lift_pmi_df) > 0:
        top_pair = lift_pmi_df.iloc[0]
        print(f"\n💡 Insight: Lift > 1 signifie association positive.")
        print(f"   {top_pair['tag_a']}+{top_pair['tag_b']} (Lift={top_pair['lift']:.2f}) "
              f"apparaissent ensemble {top_pair['lift']:.1f}x plus que par hasard!")


def plot_lift_heatmap(lift_pmi_df: pd.DataFrame,
                       priority_tags: List[str],
                       figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Visualise le Lift sous forme de heatmap.
    
    Args:
        lift_pmi_df (pd.DataFrame): DataFrame retourné par compute_lift_pmi()
        priority_tags (List[str]): Liste des tags prioritaires
        figsize (Tuple[int, int]): Taille de la figure
    """
    # Créer une matrice Lift
    lift_matrix = pd.DataFrame(1.0, index=priority_tags, columns=priority_tags)
    
    for _, row in lift_pmi_df.iterrows():
        tag_a, tag_b, lift = row['tag_a'], row['tag_b'], row['lift']
        if tag_a in priority_tags and tag_b in priority_tags:
            lift_matrix.loc[tag_a, tag_b] = lift
            lift_matrix.loc[tag_b, tag_a] = lift
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        lift_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=1.0,
        vmin=0,
        vmax=lift_matrix.max().max(),
        square=True,
        cbar_kws={'label': 'Lift (>1 = association positive)'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Heatmap de Co-occurrence (Lift)\n'
                 'Lift > 1 : Tags apparaissent ensemble plus que par hasard',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Tag', fontsize=12)
    ax.set_ylabel('Tag', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 3 : DISTRIBUTION DÉTAILLÉE DES TAGS
# ============================================================================

def analyze_tag_distribution_detailed(df: pd.DataFrame,
                                       tags_column: str = 'tags',
                                       priority_tags: Optional[List[str]] = None) -> Dict:
    """
    Analyse détaillée de la distribution des tags (avec et sans filtre prioritaire).
    
    Cette fonction fournit :
    - Nombre total de tags uniques
    - Distribution du nombre de tags par problème (min, max, mean, etc.)
    - Taux de problèmes avec au moins un tag prioritaire
    - Distribution mono-tag vs multi-tag (tous problèmes)
    - Distribution mono-tag vs multi-tag (problèmes avec tag prioritaire uniquement)
    
    Args:
        df (pd.DataFrame): DataFrame avec colonne de tags
        tags_column (str): Nom de la colonne contenant les listes de tags
        priority_tags (List[str], optional): Liste des tags prioritaires
        
    Returns:
        dict: Dictionnaire avec toutes les statistiques
    """
    # 1. Tags globaux
    all_tags = [tag for tags in df[tags_column] if isinstance(tags, list) for tag in tags]
    unique_tags = set(all_tags)
    
    # 2. Nombre de tags par problème
    df_temp = df.copy()
    df_temp['n_tags'] = df_temp[tags_column].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    tags_per_problem_stats = {
        'min': df_temp['n_tags'].min(),
        'max': df_temp['n_tags'].max(),
        'mean': df_temp['n_tags'].mean(),
        'median': df_temp['n_tags'].median(),
        'std': df_temp['n_tags'].std()
    }
    
    # 3. Problèmes avec au moins un tag prioritaire
    if priority_tags:
        df_temp['has_priority_tag'] = df_temp[tags_column].apply(
            lambda tags: any(t in priority_tags for t in tags) if isinstance(tags, list) else False
        )
        
        n_with_priority = df_temp['has_priority_tag'].sum()
        pct_with_priority = n_with_priority / len(df_temp) * 100
        
        # Filtrer sur problèmes avec tags prioritaires
        df_priority = df_temp[df_temp['has_priority_tag']].copy()
        
        # IMPORTANT: Ne garder QUE les tags prioritaires pour l'analyse
        df_priority['priority_tags_only'] = df_priority[tags_column].apply(
            lambda tags: [t for t in tags if t in priority_tags] if isinstance(tags, list) else []
        )
        
        # Recompter le nombre de tags (uniquement prioritaires)
        df_priority['n_priority_tags'] = df_priority['priority_tags_only'].apply(len)
        
        # Stats sur problèmes avec tags prioritaires (UNIQUEMENT tags prioritaires comptés)
        priority_stats = {
            'min': df_priority['n_priority_tags'].min(),
            'max': df_priority['n_priority_tags'].max(),
            'mean': df_priority['n_priority_tags'].mean(),
            'median': df_priority['n_priority_tags'].median(),
            'std': df_priority['n_priority_tags'].std()
        }
        
        # Mono-tag vs multi-tag (UNIQUEMENT tags prioritaires)
        df_priority['is_mono_tag'] = df_priority['n_priority_tags'] == 1
        n_mono_priority = df_priority['is_mono_tag'].sum()
        n_multi_priority = (~df_priority['is_mono_tag']).sum()
        
    else:
        n_with_priority = None
        pct_with_priority = None
        priority_stats = None
        n_mono_priority = None
        n_multi_priority = None
    
    # 4. Mono-tag vs multi-tag (tous problèmes)
    df_temp['is_mono_tag_all'] = df_temp['n_tags'] == 1
    n_mono_all = df_temp['is_mono_tag_all'].sum()
    n_multi_all = (~df_temp['is_mono_tag_all']).sum()
    
    return {
        'n_total_samples': len(df),
        'n_unique_tags': len(unique_tags),
        'unique_tags': sorted(unique_tags),
        'tags_per_problem': tags_per_problem_stats,
        'n_with_priority_tag': n_with_priority,
        'pct_with_priority_tag': pct_with_priority,
        'priority_tags_stats': priority_stats,
        'mono_multi_all': {
            'mono': n_mono_all,
            'multi': n_multi_all,
            'mono_pct': n_mono_all / len(df) * 100,
            'multi_pct': n_multi_all / len(df) * 100
        },
        'mono_multi_priority': {
            'mono': n_mono_priority,
            'multi': n_multi_priority,
            'mono_pct': n_mono_priority / n_with_priority * 100 if n_with_priority and n_with_priority > 0 else None,
            'multi_pct': n_multi_priority / n_with_priority * 100 if n_with_priority and n_with_priority > 0 else None
        } if priority_tags else None
    }


def display_tag_distribution_analysis(stats: Dict) -> None:
    """
    Affiche de manière formatée les résultats de l'analyse de distribution des tags.
    
    Args:
        stats (dict): Dictionnaire retourné par analyze_tag_distribution_detailed()
    """
    print("=" * 100)
    print("ANALYSE DÉTAILLÉE DE LA DISTRIBUTION DES TAGS")
    print("=" * 100)
    
    print(f"\n📊 Vue d'ensemble:")
    print(f"   Nombre total d'échantillons: {stats['n_total_samples']}")
    print(f"   Nombre de tags uniques: {stats['n_unique_tags']}")
    
    print(f"\n📈 Distribution du nombre de tags par problème (tous problèmes):")
    tpp = stats['tags_per_problem']
    print(f"   Min:    {tpp['min']}")
    print(f"   Max:    {tpp['max']}")
    print(f"   Mean:   {tpp['mean']:.2f}")
    print(f"   Median: {tpp['median']:.1f}")
    print(f"   Std:    {tpp['std']:.2f}")
    
    if stats['n_with_priority_tag'] is not None:
        print(f"\n🎯 Problèmes avec au moins un tag prioritaire:")
        print(f"   {stats['n_with_priority_tag']}/{stats['n_total_samples']} "
              f"({stats['pct_with_priority_tag']:.1f}%)")
        
        if stats['priority_tags_stats']:
            print(f"\n📈 Distribution du nombre de tags (problèmes avec tags prioritaires uniquement):")
            pts = stats['priority_tags_stats']
            print(f"   Min:    {pts['min']}")
            print(f"   Max:    {pts['max']}")
            print(f"   Mean:   {pts['mean']:.2f}")
            print(f"   Median: {pts['median']:.1f}")
            print(f"   Std:    {pts['std']:.2f}")
    
    print(f"\n🏷️  Mono-tag vs Multi-tag (tous problèmes):")
    mm_all = stats['mono_multi_all']
    print(f"   Mono-tag:  {mm_all['mono']:5d} ({mm_all['mono_pct']:.1f}%)")
    print(f"   Multi-tag: {mm_all['multi']:5d} ({mm_all['multi_pct']:.1f}%)")
    
    if stats['mono_multi_priority']:
        print(f"\n🏷️  Mono-tag vs Multi-tag (problèmes avec tags prioritaires uniquement):")
        mm_pri = stats['mono_multi_priority']
        if mm_pri['mono'] is not None:
            print(f"   Mono-tag:  {mm_pri['mono']:5d} ({mm_pri['mono_pct']:.1f}%)")
            print(f"   Multi-tag: {mm_pri['multi']:5d} ({mm_pri['multi_pct']:.1f}%)")


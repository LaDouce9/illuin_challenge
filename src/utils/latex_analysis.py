"""
Module pour l'analyse avancée du LaTeX par tag.
Calcul de l'enrichissement (Lift) des symboles LaTeX pour identifier
les symboles caractéristiques de chaque tag.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
import re


def extract_all_latex_symbols(df: pd.DataFrame, column: str = 'prob_desc_description') -> pd.DataFrame:
    """
    Extrait tous les symboles LaTeX de la colonne spécifiée.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Colonne à analyser
        
    Returns:
        pd.DataFrame: DataFrame avec une colonne par symbole LaTeX (présence binaire)
    """
    # Pattern pour extraire les commandes LaTeX
    latex_pattern = r'\\([a-zA-Z]+)'
    
    # Extraire tous les symboles uniques
    all_symbols = set()
    for text in df[column].dropna():
        symbols = re.findall(latex_pattern, str(text))
        all_symbols.update(symbols)
    
    print(f"{len(all_symbols)} symboles LaTeX uniques trouvés")
    
    # Créer un DataFrame binaire (présence/absence)
    latex_stats = {}
    for symbol in sorted(all_symbols):
        symbol_name = f'\\{symbol}'
        latex_stats[symbol_name] = df[column].apply(
            lambda x: 1 if f'\\{symbol}' in str(x) else 0
        )
    
    return pd.DataFrame(latex_stats)


def analyze_latex_enrichment_by_tag(df: pd.DataFrame, 
                                     latex_stats_df: pd.DataFrame,
                                     tags_column: str = 'tags',
                                     min_docs: int = 5) -> pd.DataFrame:
    """
    Analyse l'enrichissement des symboles LaTeX par tag.
    
    Calcule le "lift" = P(symbole|tag) / P(symbole)
    - Lift > 1 : le symbole est surreprésenté dans ce tag
    - Lift < 1 : le symbole est sous-représenté dans ce tag
    
    Args:
        df (pd.DataFrame): DataFrame source avec la colonne tags
        latex_stats_df (pd.DataFrame): DataFrame binaire (échantillons x symboles)
        tags_column (str): Nom de la colonne contenant les tags
        min_docs (int): Nombre minimum de documents avec le symbole
        
    Returns:
        pd.DataFrame: Matrice (symboles x tags) avec les scores de lift
    """
    # Filtrer les symboles trop rares
    symbol_counts = latex_stats_df.sum()
    valid_symbols = symbol_counts[symbol_counts >= min_docs].index.tolist()
    
    print(f"🔍 Analyse de l'enrichissement sur {len(valid_symbols)} symboles (présents dans ≥{min_docs} docs)")
    
    # Calculer les fréquences globales (baseline)
    global_props = (latex_stats_df[valid_symbols] > 0).mean()
    
    # Extraire tous les tags uniques
    all_tags = set()
    for tags_list in df[tags_column]:
        if isinstance(tags_list, list):
            all_tags.update(tags_list)
    all_tags = sorted(all_tags)
    
    print(f"🏷️  {len(all_tags)} tags à analyser")
    
    # Calculer le lift pour chaque tag
    lift_data = {}
    
    for tag in all_tags:
        # Indices des échantillons avec ce tag
        tag_indices = df[df[tags_column].apply(lambda x: tag in x if isinstance(x, list) else False)].index
        
        if len(tag_indices) == 0:
            continue
        
        # Fréquence du symbole dans ce tag
        tag_props = (latex_stats_df.loc[tag_indices, valid_symbols] > 0).mean()
        
        # Lift = P(symbole|tag) / P(symbole)
        # Éviter division par zéro
        lift = tag_props / global_props.replace(0, 0.001)
        
        lift_data[tag] = lift
    
    # Créer le DataFrame de lift
    lift_df = pd.DataFrame(lift_data)
    
    return lift_df


def print_tag_specific_symbols(lift_df: pd.DataFrame,
                                 priority_tags: List[str],
                                 top_n: int = 10,
                                 lift_threshold: float = 1.5) -> None:
    """
    Affiche de manière lisible les symboles caractéristiques de chaque tag.
    
    Args:
        lift_df (pd.DataFrame): DataFrame retourné par analyze_latex_enrichment_by_tag()
        priority_tags (List[str]): Liste des tags à analyser
        top_n (int): Nombre de symboles à afficher par tag
        lift_threshold (float): Seuil minimal de lift pour considérer un symbole comme "caractéristique"
    """
    print("=" * 100)
    print("SYMBOLES LaTeX CARACTÉRISTIQUES PAR TAG")
    print("=" * 100)
    
    for tag in priority_tags:
        if tag not in lift_df.columns:
            continue
        
        print(f"\nTAG: {tag.upper()}")
        print("-" * 100)
        
        # Symboles surreprésentés
        enriched = lift_df[tag][lift_df[tag] >= lift_threshold].sort_values(ascending=False).head(top_n)
        
        if len(enriched) > 0:
            print(f"  Symboles SURREPRESENTES (lift >= {lift_threshold}):")
            for i, (symbol, lift) in enumerate(enriched.items(), 1):
                print(f"    {i:2d}. {symbol:20s} -> {lift:.2f}x plus fréquent")
        else:
            print(f"  Aucun symbole fortement surreprésenté (lift >= {lift_threshold})")
        
        # Symboles sous-représentés (optionnel)
        depleted = lift_df[tag][lift_df[tag] <= 0.5].sort_values().head(5)
        if len(depleted) > 0:
            print(f"\n  Symboles SOUS-REPRESENTES (lift <= 0.5):")
            for i, (symbol, lift) in enumerate(depleted.items(), 1):
                print(f"    {i}. {symbol:20s} -> {lift:.2f}x moins fréquent")
    
    print("\n" + "=" * 100)


def plot_latex_enrichment_heatmap(lift_df: pd.DataFrame,
                                    priority_tags: Optional[List[str]] = None,
                                    top_symbols: int = 30,
                                    figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Visualise l'enrichissement des symboles LaTeX par tag sous forme de heatmap.
    
    Args:
        lift_df (pd.DataFrame): DataFrame retourné par analyze_latex_enrichment_by_tag()
        priority_tags (Optional[List[str]]): Liste des tags à afficher (None = tous)
        top_symbols (int): Nombre de symboles à afficher (les plus variables)
        figsize (Tuple[int, int]): Taille de la figure
    """
    # Filtrer les tags si nécessaire
    if priority_tags:
        available_tags = [t for t in priority_tags if t in lift_df.columns]
        if not available_tags:
            print("⚠️ Aucun tag prioritaire trouvé dans les données")
            return
        lift_subset = lift_df[available_tags]
    else:
        lift_subset = lift_df
    
    # Sélectionner les symboles les plus variables (plus intéressants)
    symbol_variance = lift_subset.var(axis=1)
    top_variable_symbols = symbol_variance.nlargest(top_symbols).index
    
    # Créer la heatmap
    heatmap_data = lift_subset.loc[top_variable_symbols].T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Utiliser une échelle centrée sur 1.0
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn',
        center=1.0,  # 1.0 = fréquence normale
        vmin=0.5,
        vmax=2.0,
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Lift (>1 = surreprésenté, <1 = sous-représenté)'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(
        'Enrichissement des symboles LaTeX par tag\n(Lift = fréquence relative vs baseline)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Symbole LaTeX', fontsize=12)
    ax.set_ylabel('Tag', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def compare_tags_latex_symbols(lift_df: pd.DataFrame,
                                 tag1: str,
                                 tag2: str,
                                 top_n: int = 15) -> None:
    """
    Compare les symboles LaTeX caractéristiques entre deux tags.
    
    Args:
        lift_df (pd.DataFrame): DataFrame retourné par analyze_latex_enrichment_by_tag()
        tag1 (str): Premier tag à comparer
        tag2 (str): Second tag à comparer
        top_n (int): Nombre de symboles à afficher
    """
    if tag1 not in lift_df.columns or tag2 not in lift_df.columns:
        print("⚠️ Un ou plusieurs tags introuvables")
        return
    
    # Calculer la différence de lift
    diff = lift_df[tag1] - lift_df[tag2]
    
    # Top symboles pour chaque tag
    top_tag1 = diff.nlargest(top_n)
    top_tag2 = diff.nsmallest(top_n)
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Symboles plus fréquents dans tag1
    ax1.barh(range(len(top_tag1)), top_tag1.values, color='steelblue')
    ax1.set_yticks(range(len(top_tag1)))
    ax1.set_yticklabels(top_tag1.index)
    ax1.set_xlabel('Différence de lift', fontsize=12)
    ax1.set_title(f'Symboles plus caractéristiques de "{tag1}"', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Symboles plus fréquents dans tag2
    ax2.barh(range(len(top_tag2)), top_tag2.values, color='coral')
    ax2.set_yticks(range(len(top_tag2)))
    ax2.set_yticklabels(top_tag2.index)
    ax2.set_xlabel('Différence de lift', fontsize=12)
    ax2.set_title(f'Symboles plus caractéristiques de "{tag2}"', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Comparaison des symboles LaTeX: {tag1} vs {tag2}', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def get_top_symbols_per_tag(lift_df: pd.DataFrame, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identifie les symboles les plus caractéristiques de chaque tag.
    
    Args:
        lift_df (pd.DataFrame): DataFrame retourné par analyze_latex_enrichment_by_tag()
        top_n (int): Nombre de symboles à retourner par tag
    
    Returns:
        Dict[str, List[Tuple[str, float]]]: Dictionnaire {tag: [(symbole, lift_score), ...]}
    """
    results = {}
    
    for tag in lift_df.columns:
        # Trier par lift décroissant
        top_symbols = lift_df[tag].sort_values(ascending=False).head(top_n)
        results[tag] = [(symbol, round(lift, 2)) for symbol, lift in top_symbols.items()]
    
    return results


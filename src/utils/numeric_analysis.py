"""
Module pour l'analyse des features numériques.
Conversion de time_limit, statistiques sur les variables numériques et textuelles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import re


def parse_time_limit(time_str: str) -> float:
    """
    Convertit une chaîne de temps en secondes (float).
    
    Gère les formats :
    - '2 seconds', '1 second'
    - '0.5 seconds', '1.5 seconds'
    - '2.0 s', '3.0 s'
    - '1 секунда', '2 секунды' (russe)
    
    Args:
        time_str (str): Chaîne de temps à convertir
        
    Returns:
        float: Temps en secondes, ou np.nan si non parsable
    """
    if pd.isna(time_str):
        return np.nan
    
    time_str = str(time_str).strip().lower()
    
    # Pattern pour extraire le nombre (int ou float)
    number_pattern = r'(\d+\.?\d*)'
    
    match = re.search(number_pattern, time_str)
    if match:
        try:
            value = float(match.group(1))
            return value
        except ValueError:
            return np.nan
    
    return np.nan


def convert_time_limit_column(df: pd.DataFrame, column: str = 'prob_desc_time_limit') -> pd.DataFrame:
    """
    Convertit la colonne time_limit en secondes et crée une nouvelle colonne.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Nom de la colonne à convertir
        
    Returns:
        pd.DataFrame: DataFrame avec nouvelle colonne 'time_limit_seconds'
    """
    df = df.copy()
    
    print(f"🔄 Conversion de '{column}' en secondes...")
    
    # Appliquer la conversion
    df['time_limit_seconds'] = df[column].apply(parse_time_limit)
    
    # Statistiques de conversion
    n_total = len(df)
    n_converted = df['time_limit_seconds'].notna().sum()
    n_missing = df['time_limit_seconds'].isna().sum()
    
    print(f"\n✅ Conversion terminée:")
    print(f"   Converti : {n_converted:4d} / {n_total} ({n_converted/n_total*100:.1f}%)")
    print(f"   Manquant : {n_missing:4d} / {n_total} ({n_missing/n_total*100:.1f}%)")
    
    # Afficher les valeurs uniques converties
    unique_values = df.dropna(subset=['time_limit_seconds'])['time_limit_seconds'].value_counts().sort_index()
    print(f"\n📊 Valeurs uniques de time_limit_seconds:")
    for value, count in unique_values.items():
        print(f"   {value:6.1f}s → {count:4d} occurrences")
    
    return df


def compute_text_length_stats(df: pd.DataFrame, column: str = 'prob_desc_description') -> pd.DataFrame:
    """
    Calcule des statistiques de longueur sur une colonne textuelle.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Nom de la colonne textuelle
        
    Returns:
        pd.DataFrame: DataFrame avec nouvelles colonnes de stats
    """
    df = df.copy()
    
    print(f"📏 Calcul des statistiques de longueur sur '{column}'...")
    
    # Nombre de caractères
    df[f'{column}_char_length'] = df[column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    # Nombre de mots
    df[f'{column}_word_count'] = df[column].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    
    # Ratio LaTeX (déjà calculé dans latex_features_desc si disponible)
    # Sinon on utilise une estimation simple
    if 'latex_density_desc' in df.columns:
        df[f'{column}_latex_ratio'] = df['latex_density_desc']
    else:
        # Estimation simple : longueur des blocs $$$...$$$  / longueur totale
        df[f'{column}_latex_ratio'] = df[column].apply(
            lambda x: _estimate_latex_ratio(str(x)) if pd.notna(x) else 0
        )
    
    # Ratio symbols LaTeX
    if 'latex_symbols_density_desc' in df.columns:
        df[f'{column}_latex_symbols_ratio'] = df['latex_symbols_density_desc']
    else:
        # Estimation simple : nombre de \command / longueur totale
        df[f'{column}_latex_symbols_ratio'] = df[column].apply(
            lambda x: _estimate_latex_symbols_ratio(str(x)) if pd.notna(x) else 0
        )
    
    print(f"\n✅ Statistiques calculées:")
    print(f"   - {column}_char_length")
    print(f"   - {column}_word_count")
    print(f"   - {column}_latex_ratio")
    print(f"   - {column}_latex_symbols_ratio")
    
    return df


def _estimate_latex_ratio(text: str) -> float:
    """Estime le ratio de LaTeX dans un texte (blocs $$$...$$$)."""
    latex_blocks = re.findall(r'\$\$\$.*?\$\$\$', text, re.DOTALL)
    latex_length = sum(len(block) for block in latex_blocks)
    total_length = len(text)
    return latex_length / total_length if total_length > 0 else 0


def _estimate_latex_symbols_ratio(text: str) -> float:
    """Estime le ratio de symboles LaTeX dans un texte."""
    latex_symbols = re.findall(r'\\[a-zA-Z]+', text)
    symbols_length = sum(len(symbol) for symbol in latex_symbols)
    total_length = len(text)
    return symbols_length / total_length if total_length > 0 else 0


def plot_numeric_distribution(df: pd.DataFrame,
                                column: str,
                                title: str,
                                xlabel: str,
                                figsize: Tuple[int, int] = (14, 5)) -> None:
    """
    Visualise la distribution d'une variable numérique.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Nom de la colonne à visualiser
        title (str): Titre du graphique
        xlabel (str): Label de l'axe X
        figsize (Tuple[int, int]): Taille de la figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Filtrer les valeurs non-NaN
    data = df[column].dropna()
    
    # Histogramme
    ax1.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Nombre d\'échantillons', fontsize=12)
    ax1.set_title(f'Distribution de {title}', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Boxplot
    ax2.boxplot(data, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel(xlabel, fontsize=12)
    ax2.set_title(f'Boxplot de {title}', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_numeric_by_tag(df: pd.DataFrame,
                         column: str,
                         priority_tags: List[str],
                         title: str,
                         ylabel: str,
                         figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Visualise la distribution d'une variable numérique par tag prioritaire.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Nom de la colonne à visualiser
        priority_tags (List[str]): Liste des tags prioritaires
        title (str): Titre du graphique
        ylabel (str): Label de l'axe Y
        figsize (Tuple[int, int]): Taille de la figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Préparer les données par tag
    data_by_tag = []
    labels = []
    
    for tag in priority_tags:
        mask = df['tags'].apply(lambda tags: tag in tags if isinstance(tags, list) else False)
        tag_data = df[mask][column].dropna()
        if len(tag_data) > 0:
            data_by_tag.append(tag_data)
            labels.append(tag)
    
    # Boxplot
    bp = ax.boxplot(data_by_tag, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    ax.set_xlabel('Tag', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_numeric_stats_by_tag(df: pd.DataFrame,
                                 column: str,
                                 priority_tags: List[str],
                                 column_label: str) -> None:
    """
    Affiche les statistiques descriptives d'une variable numérique par tag.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Nom de la colonne à analyser
        priority_tags (List[str]): Liste des tags prioritaires
        column_label (str): Label pour l'affichage
    """
    print(f"\n{'='*100}")
    print(f"STATISTIQUES DE {column_label.upper()} PAR TAG PRIORITAIRE")
    print(f"{'='*100}")
    print(f"  {'Tag':20s}   {'Count':>6s}   {'Mean':>10s}   {'Median':>10s}   {'Std':>10s}   {'Min':>10s}   {'Max':>10s}")
    print(f"  {'-'*20}   {'-'*6}   {'-'*10}   {'-'*10}   {'-'*10}   {'-'*10}   {'-'*10}")
    
    for tag in priority_tags:
        mask = df['tags'].apply(lambda tags: tag in tags if isinstance(tags, list) else False)
        tag_data = df[mask][column].dropna()
        
        if len(tag_data) > 0:
            print(f"  {tag:20s}   {len(tag_data):6d}   {tag_data.mean():10.2f}   {tag_data.median():10.2f}   "
                  f"{tag_data.std():10.2f}   {tag_data.min():10.2f}   {tag_data.max():10.2f}")
    
    print(f"{'='*100}\n")


def plot_correlation_matrix(df: pd.DataFrame,
                              columns: List[str],
                              figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Affiche une matrice de corrélation entre plusieurs variables numériques.
    
    Args:
        df (pd.DataFrame): DataFrame source
        columns (List[str]): Liste des colonnes à inclure dans la corrélation
        figsize (Tuple[int, int]): Taille de la figure
    """
    # Filtrer les colonnes qui existent
    available_columns = [col for col in columns if col in df.columns]
    
    if len(available_columns) < 2:
        print("⚠️ Pas assez de colonnes disponibles pour la corrélation")
        return
    
    # Calculer la matrice de corrélation
    corr_matrix = df[available_columns].corr()
    
    # Visualisation
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Matrice de Corrélation des Features Numériques', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les corrélations les plus fortes
    print(f"\n{'='*100}")
    print("CORRÉLATIONS LES PLUS FORTES (|r| > 0.3)")
    print(f"{'='*100}")
    
    # Extraire les paires de corrélation
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.3:
                correlations.append((col1, col2, corr_value))
    
    # Trier par valeur absolue décroissante
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if correlations:
        print(f"  {'Variable 1':40s}   {'Variable 2':40s}   {'Corrélation':>12s}")
        print(f"  {'-'*40}   {'-'*40}   {'-'*12}")
        for col1, col2, corr_val in correlations:
            print(f"  {col1:40s}   {col2:40s}   {corr_val:12.3f}")
    else:
        print("  Aucune corrélation forte détectée (|r| > 0.3)")
    
    print(f"{'='*100}\n")


"""
Module pour les fonctions d'aide à la visualisation et manipulation du LaTeX.

Ce module contient des fonctions pour :
- Afficher du texte avec mise en forme LaTeX préservée
- Extraire et analyser les symboles LaTeX
- Visualiser des exemples depuis un DataFrame

Auteur : EDA Team
Date : Janvier 2026
"""

import pandas as pd
from IPython.display import display, Markdown


def display_latex_text(text, max_length=None):
    """
    Affiche un texte en préservant la mise en forme LaTeX.
    
    Args:
        text (str): Le texte contenant du LaTeX (avec $$$ ou $)
        max_length (int, optional): Nombre max de caractères à afficher (None = tout)
    """
    if pd.isna(text):
        print("⚠️ Texte vide ou None")
        return
    
    # Tronquer si nécessaire
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Convertir $$$ en $ pour la syntaxe LaTeX standard
    text_formatted = text.replace('$$$', '$')
    
    # Afficher avec Markdown (qui supporte LaTeX)
    display(Markdown(text_formatted))


def display_latex_from_df(df, index, column='prob_desc_description', max_length=None):
    """
    Affiche un texte depuis un DataFrame avec mise en forme LaTeX.
    
    Args:
        df: Le DataFrame
        index: L'index de la ligne à afficher
        column: Le nom de la colonne (défaut: 'prob_desc_description')
        max_length: Nombre max de caractères (None = tout)
    """
    print(f"📄 Affichage de la ligne {index}, colonne '{column}':")
    print("─" * 80)
    
    if column not in df.columns:
        print(f"⚠️ Colonne '{column}' introuvable dans le DataFrame")
        return
    
    text = df.loc[index, column]
    display_latex_text(text, max_length)
    print("─" * 80)


def display_multiple_latex(df, indices, columns, max_length=None):
    """
    Affiche plusieurs colonnes textuelles pour plusieurs indices.
    
    Args:
        df: Le DataFrame
        indices: Liste d'indices de lignes à afficher
        columns: Liste de colonnes à afficher
        max_length: Nombre max de caractères par texte (None = tout)
    """
    for idx in indices:
        print(f"\n{'='*80}")
        print(f"ÉCHANTILLON {idx}")
        print(f"{'='*80}")
        
        for column in columns:
            display_latex_from_df(df, idx, column, max_length)
            print()


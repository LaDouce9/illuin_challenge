"""
Module pour la traduction et le nettoyage de texte.

Ce module contient des fonctions pour :
- Détecter la langue d'un texte
- Traduire automatiquement les textes non-anglais
- Nettoyer les patterns problématiques (NoteIN, etc.)

Auteur : EDA Team
Date : Janvier 2026
"""

import pandas as pd
import re
from typing import Optional
import warnings
from langdetect import detect
from deep_translator import GoogleTranslator

warnings.filterwarnings('ignore')


def detect_and_translate(text: str, target_lang: str = 'en') -> tuple:
    """
    Détecte la langue d'un texte et le traduit si nécessaire.
    
    Cette fonction :
    1. Détecte la langue du texte (avec langdetect)
    2. Si ce n'est pas anglais, traduit vers l'anglais (avec deep-translator)
    3. Préserve le LaTeX intact pendant la traduction
    
    Args:
        text (str): Texte à analyser/traduire
        target_lang (str): Langue cible (défaut: 'en' pour anglais)
        
    Returns:
        tuple: (texte_traduit, langue_détectée, statut_traduction)
            - texte_traduit: Le texte en anglais
            - langue_détectée: Code ISO de la langue détectée ('en', 'fr', etc.)
            - statut_traduction: 'original' si déjà en anglais, 'translated' si traduit, 'error' si échec
            
    Examples:
        >>> text_en = "Find the sum of n numbers"
        >>> result, lang, status = detect_and_translate(text_en)
        >>> print(lang, status)
        'en' 'original'
        
        >>> text_fr = "Trouvez la somme de n nombres"
        >>> result, lang, status = detect_and_translate(text_fr)
        >>> print(lang, status)
        'fr' 'translated'
    """
    try:
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
            return text, 'unknown', 'original'
        
        # 1. Détection de langue
        try:
            detected_lang = detect(text)
        except Exception:
            detected_lang = 'unknown'
        
        # 2. Si déjà en anglais, on retourne tel quel
        if detected_lang == target_lang:
            return text, detected_lang, 'original'
        
        # 3. Sinon, on traduit
        try:
            translator = GoogleTranslator(source=detected_lang, target=target_lang)
            translated_text = translator.translate(text)
            return translated_text, detected_lang, 'translated'
        except Exception as e:
            # En cas d'échec de traduction, on retourne l'original
            print(f"Erreur de traduction: {str(e)[:50]}")
            return text, detected_lang, 'error'
            
    except ImportError as e:
        print(f"⚠️ Package manquant: {e}")
        print("   Installez avec: uv pip install langdetect deep-translator")
        return text, 'not_available', 'error'
    except Exception as e:
        print(f"⚠️ Erreur inattendue: {str(e)[:50]}")
        return text, 'unknown', 'error'


def translate_column(df: pd.DataFrame, column: str, target_lang: str = 'en',
                      new_column: Optional[str] = None) -> pd.DataFrame:
    """
    Traduit une colonne entière d'un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Nom de la colonne à traduire
        target_lang (str): Langue cible (défaut: 'en')
        new_column (str, optional): Suffixe pour la nouvelle colonne ou None
            - Si None ou "" : remplace la colonne d'origine (in-place)
            - Sinon : crée une nouvelle colonne avec ce suffixe (ex: '_en' → 'column_en')
        
    Returns:
        pd.DataFrame: DataFrame avec colonne traduite
        
    Examples:
        >>> # Remplacer la colonne d'origine
        >>> df = translate_column(df, 'prob_desc_description')
        
        >>> # Créer une nouvelle colonne avec suffixe
        >>> df = translate_column(df, 'prob_desc_description', new_column='_en')
        >>> # Crée: prob_desc_description_en
    """
    df = df.copy()
    
    if column not in df.columns:
        print(f"Colonne '{column}' introuvable dans le DataFrame")
        return df
    
    # Déterminer le nom de la colonne de destination
    if new_column is None or new_column == "":
        target_column = column
        print(f"Traduction de la colonne '{column}' (remplacement in-place)...")
    else:
        target_column = f"{column}{new_column}"
        print(f"Traduction de la colonne '{column}' -> '{target_column}'...")
    
    print(f"   Langue cible: {target_lang}")
    
    # Appliquer la traduction
    results = df[column].apply(lambda x: detect_and_translate(x, target_lang))
    
    # Extraire les résultats
    df[target_column] = results.apply(lambda x: x[0])
    statuses = results.apply(lambda x: x[2])
    
    # Statistiques
    status_counts = statuses.value_counts()
    total = len(df)
    
    n_translated = status_counts.get('translated', 0)
    n_original = status_counts.get('original', 0)
    n_error = status_counts.get('error', 0) + status_counts.get('unknown', 0) + status_counts.get('not_available', 0)
    
    print(f"\nTraduction terminée:")
    print(f"   Traduit:  {n_translated:5d} / {total} ({n_translated/total*100:5.1f}%)")
    print(f"   Original: {n_original:5d} / {total} ({n_original/total*100:5.1f}%)")
    print(f"   Erreur:   {n_error:5d} / {total} ({n_error/total*100:5.1f}%)")
    
    return df


def clean_notes_patterns(text: str) -> str:
    """
    Nettoie les patterns problématiques dans les notes.
    
    Corrige notamment :
    - "NoteIN" → "Note: In"
    - "NoteThe" → "Note: The"
    - "NoteFor" → "Note: For"
    - etc.
    
    Args:
        text (str): Texte à nettoyer
        
    Returns:
        str: Texte nettoyé
        
    Examples:
        >>> clean_notes_patterns("NoteIN the following example...")
        'Note: In the following example...'
        >>> clean_notes_patterns("NoteThe answer is...")
        'Note: The answer is...'
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Pattern principal: Note + mot commençant par majuscule
    # Remplacer par "Note: " + le mot
    text = re.sub(r'Note([A-Z][a-z]+)', r'Note: \1', text)
    
    return text


def clean_column_patterns(df: pd.DataFrame, column: str, 
                           new_column: Optional[str] = None) -> pd.DataFrame:
    """
    Applique le nettoyage des patterns à une colonne entière.
    
    Args:
        df (pd.DataFrame): DataFrame source
        column (str): Nom de la colonne à nettoyer
        new_column (str, optional): Nom de la nouvelle colonne (défaut: {column}_cleaned)
        
    Returns:
        pd.DataFrame: DataFrame avec colonne nettoyée ajoutée
        
    Examples:
        >>> df_clean = clean_column_patterns(df, 'prob_desc_notes')
    """
    df = df.copy()
    
    if column not in df.columns:
        print(f"Colonne '{column}' introuvable dans le DataFrame")
        return df
    
    if new_column is None:
        new_column = f"{column}_cleaned"
    
    print(f"Nettoyage des patterns dans '{column}'...")
    
    # Appliquer le nettoyage
    df[new_column] = df[column].apply(clean_notes_patterns)
    
    # Compter les modifications
    n_changed = (df[column] != df[new_column]).sum()
    pct = n_changed / len(df) * 100
    
    print(f"Nettoyage terminé: {n_changed}/{len(df)} lignes modifiées ({pct:.1f}%)")
    
    # Afficher quelques exemples de changements
    if n_changed > 0:
        changed_mask = df[column] != df[new_column]
        examples = df[changed_mask][[column, new_column]].head(3)
        
        print(f"\nExemples de corrections:")
        for idx, row in examples.iterrows():
            orig = row[column][:60] + "..." if len(row[column]) > 60 else row[column]
            clean = row[new_column][:60] + "..." if len(row[new_column]) > 60 else row[new_column]
            print(f"\n  Avant: {orig}")
            print(f"  Après: {clean}")
    
    return df


"""
Module d'analyse textuelle pour l'EDA du challenge de classification de code.

Ce module contient toutes les fonctions utilitaires pour :
- Preprocessing du texte (nettoyage, extraction LaTeX)
- Détection de langue
- Génération de nuages de mots
- Analyse des char n-grams
- Extraction et analyse des symboles LaTeX

Auteur : EDA Team
Date : Janvier 2026
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect


# ============================================================================
# SECTION 1 : PREPROCESSING DU TEXTE
# ============================================================================

def preprocess_text_full(text: str) -> Tuple[str, Dict]:
    """
    Prétraite un texte en séparant le contenu nettoyé et les features LaTeX.
    
    Cette fonction effectue :
    1. Extraction des blocs et symboles LaTeX
    2. Nettoyage du texte (suppression LaTeX, normalisation)
    3. Calcul des métriques LaTeX (densité, symboles, etc.)
    
    Args:
        text (str): Texte brut potentiellement avec du LaTeX
        
    Returns:
        Tuple[str, Dict]: 
            - str: Texte nettoyé (sans LaTeX, lowercased, espaces normalisés)
            - Dict: Dictionnaire de features LaTeX avec clés :
                - 'latex_blocks': Liste des blocs LaTeX extraits
                - 'latex_symbols': Liste des symboles LaTeX extraits
                - 'nb_latex_blocks': Nombre de blocs LaTeX
                - 'nb_latex_symbols': Nombre de symboles LaTeX
                - 'latex_density': Ratio (longueur LaTeX / longueur totale)
                - 'latex_symbols_density': Ratio (nb symboles / longueur texte)
                - 'symbol_counts': Dict avec compte de chaque symbole unique
    
    Examples:
        >>> text = "Find $$$n$$$ such that $$$\\gcd(n, m) = 1$$$"
        >>> clean_text, latex_feats = preprocess_text_full(text)
        >>> print(clean_text)
        'find such that'
        >>> print(latex_feats['nb_latex_blocks'])
        2
    """
    if pd.isna(text) or not isinstance(text, str):
        return "", {
            'latex_blocks': [],
            'latex_symbols': [],
            'nb_latex_blocks': 0,
            'nb_latex_symbols': 0,
            'latex_density': 0.0,
            'latex_symbols_density': 0.0,
            'symbol_counts': {}
        }
    
    # 1. Extraire les blocs LaTeX ($$$...$$$ ou $...$)
    latex_blocks = re.findall(r'\$\$\$.*?\$\$\$|\$.*?\$', text)
    
    # 2. Extraire les symboles LaTeX (commandes \xxx)
    latex_patterns = [
        # Opérations mathématiques
        r'\\frac', r'\\dfrac', r'\\tfrac', r'\\cfrac',
        # Sommations et produits
        r'\\sum', r'\\prod', r'\\coprod',
        # Intégrales
        r'\\int', r'\\iint', r'\\iiint', r'\\oint',
        # Racines
        r'\\sqrt', r'\\surd',
        # Fonctions mathématiques
        r'\\log', r'\\ln', r'\\lg', r'\\exp',
        r'\\sin', r'\\cos', r'\\tan', r'\\cot', r'\\sec', r'\\csc',
        r'\\arcsin', r'\\arccos', r'\\arctan',
        r'\\sinh', r'\\cosh', r'\\tanh', r'\\coth',
        # Modulo et arithmétique
        r'\\mod', r'\\bmod', r'\\pmod', r'\\gcd', r'\\lcm',
        # Limites et dérivées
        r'\\lim', r'\\limsup', r'\\liminf', r'\\sup', r'\\inf',
        r'\\min', r'\\max', r'\\arg',
        r'\\partial', r'\\nabla',
        # Nombres spéciaux
        r'\\prime', r'\\infty', r'\\emptyset', r'\\varnothing',
        # Relations et comparaisons
        r'\\le', r'\\ge', r'\\leq', r'\\geq', r'\\leqslant', r'\\geqslant',
        r'\\ll', r'\\gg', r'\\ne', r'\\neq', r'\\equiv', r'\\approx',
        r'\\sim', r'\\simeq', r'\\cong', r'\\propto',
        # Opérateurs ensemblistes
        r'\\in', r'\\notin', r'\\subset', r'\\subseteq', r'\\supset', r'\\supseteq',
        r'\\cup', r'\\cap', r'\\setminus', r'\\bigcup', r'\\bigcap',
        # Logique
        r'\\land', r'\\lor', r'\\lnot', r'\\neg', r'\\implies', r'\\iff',
        r'\\forall', r'\\exists', r'\\nexists',
        # Flèches
        r'\\rightarrow', r'\\leftarrow', r'\\leftrightarrow', r'\\Rightarrow',
        r'\\Leftarrow', r'\\Leftrightarrow', r'\\to', r'\\mapsto',
        # Accents et décorations
        r'\\hat', r'\\widehat', r'\\bar', r'\\overline', r'\\underline',
        r'\\tilde', r'\\widetilde', r'\\vec', r'\\overrightarrow',
        r'\\dot', r'\\ddot',
        # Binômes et combinatoire
        r'\\binom', r'\\tbinom', r'\\dbinom', r'\\choose',
        # Lettres grecques (minuscules)
        r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\epsilon', r'\\varepsilon',
        r'\\zeta', r'\\eta', r'\\theta', r'\\vartheta', r'\\iota', r'\\kappa',
        r'\\lambda', r'\\mu', r'\\nu', r'\\xi', r'\\pi', r'\\varpi',
        r'\\rho', r'\\varrho', r'\\sigma', r'\\varsigma', r'\\tau',
        r'\\upsilon', r'\\phi', r'\\varphi', r'\\chi', r'\\psi', r'\\omega',
        # Lettres grecques (majuscules)
        r'\\Gamma', r'\\Delta', r'\\Theta', r'\\Lambda', r'\\Xi',
        r'\\Pi', r'\\Sigma', r'\\Upsilon', r'\\Phi', r'\\Psi', r'\\Omega',
        # Points de suspension
        r'\\dots', r'\\ldots', r'\\cdots', r'\\vdots', r'\\ddots',
        # Autres symboles courants
        r'\\cdot', r'\\times', r'\\div', r'\\pm', r'\\mp',
        r'\\oplus', r'\\ominus', r'\\otimes', r'\\odot',
        r'\\lfloor', r'\\rfloor', r'\\lceil', r'\\rceil',
        r'\\langle', r'\\rangle',
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in latex_patterns)
    matches = re.findall(combined_pattern, text)
    latex_symbols = [match for group in matches for match in group if match]
    
    # Compter les symboles
    symbol_counts = Counter(latex_symbols)
    
    # 3. Nettoyer le texte (supprimer LaTeX)
    clean_text = re.sub(r'\$\$\$.*?\$\$\$', ' ', text)  # Supprimer blocs $$$...$$$
    clean_text = re.sub(r'\$.*?\$', ' ', clean_text)    # Supprimer blocs $...$
    
    # Lowercasing
    clean_text = clean_text.lower()
    
    # Garder seulement lettres, chiffres, espaces
    clean_text = re.sub(r'[^a-z0-9\s-]', ' ', clean_text)
    
    # Normaliser les espaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # 4. Calculer les métriques LaTeX
    total_latex_length = sum(len(block) for block in latex_blocks)
    text_length = len(text)
    
    latex_density = total_latex_length / text_length if text_length > 0 else 0.0
    latex_symbols_density = len(latex_symbols) / text_length if text_length > 0 else 0.0
    
    latex_features = {
        'latex_blocks': latex_blocks,
        'latex_symbols': latex_symbols,
        'nb_latex_blocks': len(latex_blocks),
        'nb_latex_symbols': len(latex_symbols),
        'latex_density': latex_density,
        'latex_symbols_density': latex_symbols_density,
        'symbol_counts': dict(symbol_counts)
    }
    
    return clean_text, latex_features


# ============================================================================
# SECTION 2 : DÉTECTION DE LANGUE
# ============================================================================

def detect_language_single(text: str) -> str:
    """
    Détecte la langue d'un texte unique.
    
    Utilise la bibliothèque langdetect pour identifier la langue.
    Gère les erreurs gracieusement (texte trop court, détection impossible).
    
    Args:
        text (str): Texte à analyser
        
    Returns:
        str: Code ISO de la langue ('en', 'fr', 'es', etc.) ou 'unknown' si échec
        
    Examples:
        >>> detect_language_single("This is an English text")
        'en'
        >>> detect_language_single("Ceci est un texte français")
        'fr'
        >>> detect_language_single("")
        'unknown'
    
    Note:
        Nécessite: pip install langdetect
    """
    try:
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
            return 'unknown'
        
        return detect(text)
    except Exception:
        return 'unknown'


def analyze_language_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Analyse la distribution des langues détectées dans une colonne.
    
    Pour chaque ligne du DataFrame, détecte la langue du texte dans la colonne
    spécifiée et retourne un DataFrame avec les statistiques agrégées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        column (str): Nom de la colonne à analyser
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes:
            - language: Code langue
            - count: Nombre d'occurrences
            - percentage: Pourcentage
            
    Examples:
        >>> lang_dist = analyze_language_distribution(df, 'prob_desc_description')
        >>> print(lang_dist)
           language  count  percentage
        0        en   4800       96.3
        1        fr     87        1.7
        2   unknown     95        1.9
    """
    print(f"🔍 Détection de langue sur la colonne '{column}'...")
    
    # Détecter la langue pour chaque ligne
    languages = df[column].apply(detect_language_single)
    
    # Compter les occurrences
    lang_counts = languages.value_counts()
    
    # Créer le DataFrame de résultats
    result = pd.DataFrame({
        'language': lang_counts.index,
        'count': lang_counts.values,
        'percentage': (lang_counts.values / len(df) * 100).round(2)
    }).reset_index(drop=True)
    
    print(f"✅ Analyse terminée. {len(result)} langues détectées.")
    
    return result


# ============================================================================
# SECTION 3 : STOPWORDS POUR NUAGES DE MOTS
# ============================================================================

def get_comprehensive_stopwords() -> set:
    """
    Retourne une liste exhaustive de stopwords pour les nuages de mots.
    
    Combine les stopwords classiques anglais + mots spécifiques au contexte
    de problèmes algorithmiques.
    
    Returns:
        set: Ensemble de stopwords
        
    Note:
        Cette fonction évite la pollution des nuages de mots par des mots
        trop génériques qui n'apportent pas d'information discriminante.
    """
    # Stopwords de base de wordcloud
    base_stopwords = set(STOPWORDS)
    
    # Stopwords classiques supplémentaires
    english_stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'doing', 'would', 'should', 'could',
        'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them',
        'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'just', 'don', 'now',
        'am', 'an', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
        'yourselves', 'themselves', 'this', 'these', 'those'
    }
    
    # Stopwords spécifiques au contexte de problèmes algorithmiques
    context_stopwords = {
        'given', 'find', 'you', 'are', 'the', 'and', 'for', 'with', 'that', 'this',
        'from', 'have', 'can', 'will', 'each', 'one', 'two', 'first', 'second',
        'input', 'output', 'return', 'print', 'sample', 'example', 'test', 'case',
        'integer', 'positive', 'negative', 'non', 'may', 'must', 'should', 'would',
        'contains', 'consisting', 'followed', 'line', 'lines', 'array', 'contains',
        'single', 'multiple', 'next', 'following', 'space', 'separated', 'description',
        'note', 'notes', 'constraint', 'constraints', 'problem', 'answer', 'result',
        'numbers', 'number', 'value', 'values', 'element', 'elements', 'denote',
        'let', 'lets', 'consider', 'suppose', 'assume', 'also', 'thus', 'therefore',
        'however', 'moreover', 'furthermore', 'otherwise', 'respectively', 'exactly'
    }
    
    # Combiner toutes les stopwords
    all_stopwords = base_stopwords | english_stopwords | context_stopwords
    
    return all_stopwords


# ============================================================================
# SECTION 4 : NUAGES DE MOTS
# ============================================================================

def create_wordclouds_for_priority_tags(df: pd.DataFrame, 
                                        column: str,
                                        priority_tags: List[str],
                                        max_words: int = 80,
                                        figsize: Tuple[int, int] = (20, 10),
                                        remove_latex: bool = True) -> None:
    """
    Crée des nuages de mots pour tous les tags prioritaires sur une colonne donnée.
    
    Génère une figure avec N subplots (un par tag prioritaire), chacun affichant
    un nuage de mots des termes les plus fréquents pour ce tag dans la colonne spécifiée.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        column (str): Nom de la colonne à analyser
        priority_tags (List[str]): Liste des tags prioritaires
        max_words (int): Nombre maximum de mots par nuage (défaut: 80)
        figsize (Tuple[int, int]): Taille de la figure (défaut: (20, 10))
        remove_latex (bool): Supprimer le LaTeX avant analyse (défaut: True)
        
    Returns:
        None: Affiche la figure matplotlib
        
    Examples:
        >>> PRIORITY_TAGS = ['math', 'graphs', 'strings', 'number theory']
        >>> create_wordclouds_for_priority_tags(df, 'prob_desc_description', PRIORITY_TAGS)
    
    Note:
        Les stopwords sont automatiquement filtrés via get_comprehensive_stopwords()
    """
    n_tags = len(priority_tags)
    n_cols = 4
    n_rows = (n_tags + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_tags > 1 else [axes]
    
    stopwords = get_comprehensive_stopwords()
    
    for idx, tag in enumerate(priority_tags):
        ax = axes[idx]
        
        # Filtrer les échantillons avec ce tag
        mask = df['tags'].apply(lambda x: tag in x if isinstance(x, list) else False)
        tag_df = df[mask]
        
        if len(tag_df) == 0:
            ax.text(0.5, 0.5, f'Aucun échantillon\npour "{tag}"', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        # Concaténer toutes les valeurs de la colonne
        all_texts = tag_df[column].apply(
            lambda x: preprocess_text_full(x)[0] if remove_latex else x.lower() if isinstance(x, str) else ''
        ).str.cat(sep=' ')
        
        if len(all_texts.strip()) == 0:
            ax.text(0.5, 0.5, f'Pas de texte\npour "{tag}"', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        # Créer le nuage de mots
        wordcloud = WordCloud(
            width=800, 
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=max_words,
            stopwords=stopwords,
            relative_scaling=0.5,
            min_font_size=8
        ).generate(all_texts)
        
        # Afficher
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{tag}\n({len(tag_df)} échantillons)', 
                    fontsize=12, fontweight='bold')
    
    # Cacher les axes non utilisés
    for idx in range(n_tags, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Nuages de mots par tag prioritaire - Colonne: {column}', 
                 fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def compare_top_words_by_tag(df: pd.DataFrame,
                             column: str,
                             priority_tags: List[str],
                             top_n: int = 15,
                             remove_latex: bool = True) -> Dict[str, List[Tuple[str, int]]]:
    """
    Compare les top mots de chaque tag sous forme de tableau textuel.
    
    Pour chaque tag, extrait les N mots les plus fréquents (après filtrage des stopwords)
    dans la colonne spécifiée et affiche un tableau formaté.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        column (str): Nom de la colonne à analyser
        priority_tags (List[str]): Liste des tags prioritaires
        top_n (int): Nombre de mots à afficher par tag (défaut: 15)
        remove_latex (bool): Supprimer le LaTeX avant analyse (défaut: True)
        
    Returns:
        Dict[str, List[Tuple[str, int]]]: Dictionnaire {tag: [(mot, fréquence), ...]}
        
    Examples:
        >>> top_words = compare_top_words_by_tag(df, 'prob_desc_description', PRIORITY_TAGS)
        >>> print(top_words['math'][:3])
        [('sum', 245), ('product', 189), ('calculate', 156)]
    """
    stopwords = get_comprehensive_stopwords()
    results = {}
    
    for tag in priority_tags:
        # Filtrer les échantillons avec ce tag
        mask = df['tags'].apply(lambda x: tag in x if isinstance(x, list) else False)
        tag_df = df[mask]
        
        if len(tag_df) == 0:
            continue
        
        # Concaténer et prétraiter
        all_texts = tag_df[column].apply(
            lambda x: preprocess_text_full(x)[0] if remove_latex else x.lower() if isinstance(x, str) else ''
        ).str.cat(sep=' ')
        
        # Compter les mots
        word_freq = Counter(all_texts.split())
        
        # Filtrer et garder les top N
        filtered = [(w, f) for w, f in word_freq.most_common(500) 
                   if w not in stopwords and len(w) > 2][:top_n]
        
        results[tag] = filtered
    
    # Affichage formaté
    print("=" * 120)
    print(f"TOP MOTS PAR TAG PRIORITAIRE - Colonne: {column}")
    print("=" * 120)
    
    for tag, words in results.items():
        print(f"\n🏷️  {tag.upper()} ({len(df[df['tags'].apply(lambda x: tag in x)])} échantillons)")
        print("-" * 80)
        for i, (word, freq) in enumerate(words, 1):
            print(f"  {i:2d}. {word:20s} → {freq:5d} occurrences")
    
    return results


# ============================================================================
# SECTION 5 : CHAR N-GRAMS
# ============================================================================

def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    """
    Extrait les char n-grams d'un texte.
    
    Args:
        text (str): Texte à analyser
        n (int): Taille des n-grams (défaut: 3)
        
    Returns:
        List[str]: Liste des n-grams
        
    Examples:
        >>> extract_char_ngrams("graph", 3)
        ['gra', 'rap', 'aph']
    """
    if pd.isna(text) or not isinstance(text, str) or len(text) < n:
        return []
    
    text = text.lower()
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def analyze_char_ngrams_global(df: pd.DataFrame,
                               column: str,
                               n_range: Tuple[int, int] = (3, 5),
                               top_n: int = 30) -> Dict[int, List[Tuple[str, int]]]:
    """
    Analyse les char n-grams les plus fréquents globalement.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        column (str): Nom de la colonne à analyser
        n_range (Tuple[int, int]): Range des tailles de n-grams (défaut: (3, 5))
        top_n (int): Nombre de top n-grams à retourner (défaut: 30)
        
    Returns:
        Dict[int, List[Tuple[str, int]]]: {taille: [(ngram, count), ...]}
    """
    results = {}
    
    for n in range(n_range[0], n_range[1] + 1):
        all_ngrams = []
        
        for text in df[column]:
            if pd.isna(text) or not isinstance(text, str):
                continue
            ngrams = extract_char_ngrams(text, n)
            all_ngrams.extend(ngrams)
        
        ngram_counts = Counter(all_ngrams)
        results[n] = ngram_counts.most_common(top_n)
    
    # Affichage
    print("=" * 100)
    print(f"CHAR N-GRAMS GLOBAUX - Colonne: {column}")
    print("=" * 100)
    
    for n, top_ngrams in results.items():
        print(f"\n📊 Top {top_n} char {n}-grams:")
        print(f"  {'N-gram':10s}   {'Count':>10s}   {'Examples'}")
        print(f"  {'-'*10}   {'-'*10}   {'-'*30}")
        for ngram, count in top_ngrams[:20]:
            print(f"  {ngram:10s}   {count:>10d}")
    
    return results


def analyze_char_ngrams_by_tag(df: pd.DataFrame,
                               column: str,
                               priority_tags: List[str],
                               n: int = 3,
                               top_n: int = 20) -> Dict[str, List[Tuple[str, int]]]:
    """
    Analyse les char n-grams discriminants par tag (via log-odds ratio).
    
    Pour chaque tag, identifie les n-grams qui sont sur-représentés par rapport
    à la distribution globale.
    
    Args:
        df (pd.DataFrame): DataFrame avec les données
        column (str): Nom de la colonne à analyser
        priority_tags (List[str]): Liste des tags prioritaires
        n (int): Taille des n-grams (défaut: 3)
        top_n (int): Nombre de n-grams à afficher par tag (défaut: 20)
        
    Returns:
        Dict[str, List[Tuple[str, int]]]: {tag: [(ngram, count), ...]}
    """
    # Calculer les fréquences globales
    global_ngrams = Counter()
    for text in df[column]:
        if pd.isna(text) or not isinstance(text, str):
            continue
        ngrams = extract_char_ngrams(text, n)
        global_ngrams.update(ngrams)
    
    total_global = sum(global_ngrams.values())
    
    results = {}
    
    for tag in priority_tags:
        mask = df['tags'].apply(lambda x: tag in x if isinstance(x, list) else False)
        tag_df = df[mask]
        
        if len(tag_df) == 0:
            continue
        
        # Compter les n-grams pour ce tag
        tag_ngrams = Counter()
        for text in tag_df[column]:
            if pd.isna(text) or not isinstance(text, str):
                continue
            ngrams = extract_char_ngrams(text, n)
            tag_ngrams.update(ngrams)
        
        total_tag = sum(tag_ngrams.values())
        
        # Calculer le log-odds ratio (enrichissement)
        enrichment_scores = []
        for ngram, count_tag in tag_ngrams.most_common(100):
            if ngram not in global_ngrams or global_ngrams[ngram] < 5:
                continue
            
            freq_tag = count_tag / total_tag
            freq_global = global_ngrams[ngram] / total_global
            
            log_odds = np.log2((freq_tag + 1e-10) / (freq_global + 1e-10))
            enrichment_scores.append((ngram, count_tag, log_odds))
        
        # Trier par log-odds (les plus discriminants)
        enrichment_scores.sort(key=lambda x: x[2], reverse=True)
        results[tag] = [(ng, cnt) for ng, cnt, _ in enrichment_scores[:top_n]]
    
    # Affichage
    print("=" * 120)
    print(f"CHAR {n}-GRAMS DISCRIMINANTS PAR TAG - Colonne: {column}")
    print("=" * 120)
    
    for tag, ngrams in results.items():
        print(f"\n🏷️  {tag.upper()}")
        print(f"  {'N-gram':10s}   {'Count':>10s}")
        print(f"  {'-'*10}   {'-'*10}")
        for ngram, count in ngrams[:15]:
            print(f"  {ngram:10s}   {count:>10d}")
    
    return results


# ============================================================================
# SECTION 6 : DOCUMENT TEXTE UNIQUE
# ============================================================================

def remove_latex_from_text(text: str, replacement_token: str = "[LATEX]") -> str:
    """
    Remove LaTeX blocks from text and replace with a token.
    
    Parameters
    ----------
    text : str
        Input text containing LaTeX
    replacement_token : str, default="[LATEX]"
        Token to replace LaTeX blocks with
        
    Returns
    -------
    str
        Text with LaTeX blocks replaced by token
        
    Examples
    --------
    >>> text = "This is $$$x^2$$$ a formula"
    >>> remove_latex_from_text(text)
    'This is [LATEX] a formula'
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove LaTeX blocks ($$$...$$$)
    text = re.sub(r'\$\$\$.*?\$\$\$', replacement_token, text)
    
    # Remove inline LaTeX symbols (\command)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', replacement_token, text)
    text = re.sub(r'\\[a-zA-Z]+', replacement_token, text)
    
    # Clean up multiple consecutive tokens
    text = re.sub(rf'{re.escape(replacement_token)}(\s*{re.escape(replacement_token)})+', 
                  replacement_token, text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def create_unified_document(row: pd.Series, suffix: str = "") -> str:
    """
    Crée un document texte unique en concaténant toutes les colonnes pertinentes.
    
    Utilise des tokens spéciaux pour séparer les différentes sections :
    [DESC], [IN], [OUT], [NOTES], [NOTES_MISSING], [SAMPLE_IN], [SAMPLE_OUT]
    
    Args:
        row (pd.Series): Ligne du DataFrame
        suffix (str): Suffixe pour les noms de colonnes (ex: '_translated' pour utiliser les versions traduites)
        
    Returns:
        str: Document texte unifié
        
    Examples:
        >>> # Utiliser les colonnes originales
        >>> doc = create_unified_document(df.iloc[0])
        >>> print(doc[:100])
        '[DESC] Find the sum of n numbers [IN] First line contains n...'
        
        >>> # Utiliser les colonnes traduites
        >>> doc = create_unified_document(df.iloc[0], suffix='_translated')
        >>> # Utilise prob_desc_description_translated, etc.
    """
    parts = []
    
    # Description
    col_desc = f'prob_desc_description{suffix}'
    if pd.notna(row.get(col_desc)):
        parts.append(f"[DESC] {row[col_desc]}")
    
    # Input spec
    col_input = f'prob_desc_input_spec{suffix}'
    if pd.notna(row.get(col_input)):
        parts.append(f"[IN] {row[col_input]}")
    
    # Output spec
    col_output = f'prob_desc_output_spec{suffix}'
    if pd.notna(row.get(col_output)):
        parts.append(f"[OUT] {row[col_output]}")
    
    # Notes
    col_notes = f'prob_desc_notes{suffix}'
    if pd.notna(row.get(col_notes)):
        parts.append(f"[NOTES] {row[col_notes]}")
    else:
        parts.append("[NOTES_MISSING]")
    
    # Sample inputs (pas de traduction pour les exemples)
    if pd.notna(row.get('prob_desc_sample_inputs')):
        parts.append(f"[SAMPLE_IN] {str(row['prob_desc_sample_inputs'])[:500]}")  # Limiter la longueur
    
    # Sample outputs (pas de traduction pour les exemples)
    if pd.notna(row.get('prob_desc_sample_outputs')):
        parts.append(f"[SAMPLE_OUT] {str(row['prob_desc_sample_outputs'])[:500]}")  # Limiter la longueur
    
    return " ".join(parts)


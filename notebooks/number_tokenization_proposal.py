"""
Proposition de Tokenisation Semantique des Nombres pour TF-IDF V2
================================================================

Ce script propose une fonction pour remplacer les nombres par des tokens
semantiques afin de reduire l'explosion du vocabulaire.
"""

import re
from typing import Dict, Set

# Configuration: nombres a conserver tels quels
PRESERVE_NUMBERS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
PRESERVE_VARS = {'n', 'm', 'k', 'x', 'y', 'z', 'i', 'j', 'l', 'r'}  # Variables algorithmiques courantes

# Patterns de reconnaissance
MODULO_CONSTANTS = {
    '1000000007', '1000000009', '998244353',  # Modulos classiques
    '109', '107'  # Notations courtes pour 10^9+7, etc.
}

def tokenize_numbers(text: str) -> str:
    """
    Remplace les nombres par des tokens semantiques.
    
    Ordre de traitement (important) :
    1. Modulo constants (1000000007 -> NUM_MOD)
    2. Notation exponentielle (1e9, 10^9 -> NUM_EXP)
    3. Floats (3.14 -> NUM_FLOAT)
    4. Binaires purs (101010 -> NUM_BIN si pertinent)
    5. Entiers restants (> 10 -> NUM_INT)
    6. Preserver 0-10 et variables n, m, k
    
    Parameters
    ----------
    text : str
        Texte nettoye (apres clean_text)
        
    Returns
    -------
    str
        Texte avec nombres tokenises
        
    Examples
    --------
    >>> tokenize_numbers("array of 1000 elements modulo 1000000007")
    "array of NUM_INT elements modulo NUM_MOD"
    
    >>> tokenize_numbers("complexity is 1e9 operations")
    "complexity is NUM_EXP operations"
    
    >>> tokenize_numbers("values 0 1 2 are special")
    "values 0 1 2 are special"  # Preserves
    """
    
    # Etape 1: Remplacer les constantes modulo
    for mod_const in MODULO_CONSTANTS:
        text = re.sub(rf'\b{mod_const}\b', ' NUM_MOD ', text)
    
    # Etape 2: Notation exponentielle (1e9, 2e5, 10e6, 10^9, 2^10)
    # Format: Xe[+-]?Y ou X^Y
    text = re.sub(r'\b\d+e[+-]?\d+\b', ' NUM_EXP ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\^\d+\b', ' NUM_EXP ', text)
    text = re.sub(r'\b10\*\*\d+\b', ' NUM_EXP ', text)  # 10**9
    
    # Etape 3: Floats (nombres avec point decimal)
    # Ex: 3.14, 0.5, 2.718
    text = re.sub(r'\b\d+\.\d+\b', ' NUM_FLOAT ', text)
    
    # Etape 4: Binaires (sequences de 0 et 1 de longueur > 4)
    # Note: optionnel, peut etre trop agressif
    # Uncomment si pertinent:
    # text = re.sub(r'\b[01]{5,}\b', ' NUM_BIN ', text)
    
    # Etape 5: Entiers > 10 (en preservant 0-10)
    def replace_integer(match):
        num_str = match.group(0)
        
        # Preserver 0-10
        if num_str in PRESERVE_NUMBERS:
            return num_str
        
        # Preserver variables courantes (n, m, k, etc.)
        if num_str in PRESERVE_VARS:
            return num_str
        
        # Preserver patterns n1, n2, k1, etc.
        if re.match(r'^[nmkxyzijlr]\d+$', num_str):
            return num_str
        
        # Remplacer les autres entiers
        return ' NUM_INT '
    
    # Matcher tous les entiers
    text = re.sub(r'\b\d+\b', replace_integer, text)
    
    # Normaliser les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def analyze_number_impact(df, text_column='unified_document_clean'):
    """
    Analyse l'impact de la tokenisation des nombres sur le vocabulaire.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe avec la colonne de texte
    text_column : str
        Nom de la colonne de texte
        
    Returns
    -------
    dict
        Statistiques avant/apres tokenisation
    """
    from collections import Counter
    import pandas as pd
    
    # Vocabulaire AVANT tokenisation
    all_tokens_before = []
    for text in df[text_column]:
        tokens = re.findall(r'\b\w+\b', str(text).lower())
        all_tokens_before.extend(tokens)
    
    vocab_before = Counter(all_tokens_before)
    
    # Identifier les nombres dans le vocabulaire
    number_tokens = {token for token in vocab_before.keys() if re.match(r'^\d+$', token)}
    number_tokens_count = len(number_tokens)
    number_occurrences = sum(vocab_before[token] for token in number_tokens)
    
    # Appliquer tokenisation
    df_temp = df.copy()
    df_temp[f'{text_column}_numtok'] = df_temp[text_column].apply(tokenize_numbers)
    
    # Vocabulaire APRES tokenisation
    all_tokens_after = []
    for text in df_temp[f'{text_column}_numtok']:
        tokens = re.findall(r'\b\w+\b', str(text).lower())
        all_tokens_after.extend(tokens)
    
    vocab_after = Counter(all_tokens_after)
    
    # Compter les nouveaux tokens NUM_*
    num_tokens_added = {
        'NUM_INT': vocab_after.get('num_int', 0),
        'NUM_FLOAT': vocab_after.get('num_float', 0),
        'NUM_EXP': vocab_after.get('num_exp', 0),
        'NUM_MOD': vocab_after.get('num_mod', 0),
        'NUM_BIN': vocab_after.get('num_bin', 0)
    }
    
    return {
        'vocab_size_before': len(vocab_before),
        'vocab_size_after': len(vocab_after),
        'reduction': len(vocab_before) - len(vocab_after),
        'reduction_pct': (1 - len(vocab_after) / len(vocab_before)) * 100,
        'number_tokens_removed': number_tokens_count,
        'number_occurrences': number_occurrences,
        'num_tokens_added': num_tokens_added,
        'top_numbers_removed': sorted(
            [(token, count) for token, count in vocab_before.items() if token in number_tokens],
            key=lambda x: x[1],
            reverse=True
        )[:20]
    }


def print_tokenization_examples():
    """Affiche des exemples de tokenisation"""
    examples = [
        "find the sum of n numbers from 1 to 1000",
        "modulo 1000000007 is used for large numbers",
        "complexity is O(n) where n can be up to 1e9",
        "array has 100000 elements",
        "probability is 0.5 or 3.14159",
        "binary string 101010111000",
        "for i from 0 to 10",
        "n1 and n2 are inputs",
        "calculate 2^10 + 10^9"
    ]
    
    print("="*80)
    print("EXEMPLES DE TOKENISATION DES NOMBRES")
    print("="*80)
    
    for i, example in enumerate(examples, 1):
        result = tokenize_numbers(example)
        print(f"\n{i}. Original:")
        print(f"   {example}")
        print(f"   Tokenise:")
        print(f"   {result}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Tester la fonction
    print_tokenization_examples()
    
    print("\nCONFIGURATION:")
    print(f"  Nombres preserves (0-10): {PRESERVE_NUMBERS}")
    print(f"  Variables preservees: {PRESERVE_VARS}")
    print(f"  Constantes modulo: {MODULO_CONSTANTS}")


"""
Advanced Multi-Label Analysis & Code Pattern Detection
Based on ChatGPT recommendations
"""
import sys
sys.path.append('/app')

import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import re
import json

print("=" * 80)
print("ANALYSES AVANCÉES - MULTI-LABEL & CODE PATTERNS")
print("=" * 80)

# Load validated dataset
df = pd.read_parquet('/app/data/processed/dataset_validated.parquet')
PRIORITY_TAGS = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']

# ============================================================================
# 1. CO-OCCURRENCE AVANCÉE: LIFT & PMI
# ============================================================================
print("\n[1/5] Calcul Lift & PMI pour co-occurrences...")

# Calculer P(A), P(B), P(A,B) pour tous les tags prioritaires
tag_probs = {}
for tag in PRIORITY_TAGS:
    count = sum(1 for tags in df['tags'] if tag in tags)
    tag_probs[tag] = count / len(df)

# Co-occurrence avec lift
cooccurrence_lift = {}
for tag_a, tag_b in combinations(PRIORITY_TAGS, 2):
    # P(A and B)
    count_both = sum(1 for tags in df['tags'] if tag_a in tags and tag_b in tags)
    p_both = count_both / len(df)
    
    # Lift = P(A,B) / (P(A) * P(B))
    lift = p_both / (tag_probs[tag_a] * tag_probs[tag_b]) if tag_probs[tag_a] * tag_probs[tag_b] > 0 else 0
    
    # PMI = log(P(A,B) / (P(A) * P(B)))
    pmi = np.log(lift) if lift > 0 else 0
    
    # P(B|A) et P(A|B)
    p_b_given_a = count_both / (tag_probs[tag_a] * len(df)) if tag_probs[tag_a] > 0 else 0
    p_a_given_b = count_both / (tag_probs[tag_b] * len(df)) if tag_probs[tag_b] > 0 else 0
    
    cooccurrence_lift[(tag_a, tag_b)] = {
        'count': count_both,
        'lift': lift,
        'pmi': pmi,
        'p_b_given_a': p_b_given_a,
        'p_a_given_b': p_a_given_b
    }

# Top paires par lift
sorted_by_lift = sorted(cooccurrence_lift.items(), key=lambda x: x[1]['lift'], reverse=True)

print(f"✓ Top 10 paires par Lift (association forte):")
for (tag_a, tag_b), metrics in sorted_by_lift[:10]:
    print(f"  - {tag_a:15s} + {tag_b:15s}: Lift={metrics['lift']:.2f}, PMI={metrics['pmi']:.2f}, Count={metrics['count']}")

# ============================================================================
# 2. CODE PATTERNS: IMPORTS & STRUCTURES
# ============================================================================
print("\n[2/5] Détection de patterns dans le code...")

# Imports communs
COMMON_IMPORTS = [
    'collections', 'heapq', 'bisect', 'itertools', 'functools',
    'math', 'sys', 'random', 're', 'string', 'operator'
]

def detect_imports(code):
    """Detect imports in code"""
    imports = {}
    for imp in COMMON_IMPORTS:
        # Match "import X" or "from X import"
        pattern = rf'\b(import\s+{imp}|from\s+{imp}\s+import)\b'
        imports[f'import_{imp}'] = 1 if re.search(pattern, code) else 0
    return imports

# Patterns algorithmiques
def detect_algo_patterns(code):
    """Detect algorithmic patterns"""
    patterns = {}
    
    # BFS/DFS indicators
    patterns['has_deque'] = 1 if 'deque' in code else 0
    patterns['has_queue'] = 1 if ('Queue' in code or 'deque' in code) else 0
    patterns['has_stack'] = 1 if 'stack' in code.lower() or 'append' in code and 'pop' in code else 0
    
    # DSU/Union-Find
    patterns['has_dsu'] = 1 if ('parent' in code and 'find' in code) or 'union' in code.lower() else 0
    
    # Recursion
    patterns['has_recursion'] = 1 if 'setrecursionlimit' in code else 0
    
    # DP indicators
    patterns['has_dp'] = 1 if ('dp[' in code or 'memo' in code.lower()) else 0
    
    # Graph adjacency
    patterns['has_adjacency'] = 1 if ('adj' in code.lower() or 'graph[' in code or 'edges' in code) else 0
    
    # Sorting
    patterns['has_sort'] = 1 if '.sort' in code or 'sorted(' in code else 0
    
    # Binary search
    patterns['has_bisect'] = 1 if 'bisect' in code else 0
    
    return patterns

# Appliquer sur un échantillon (pour performance)
sample_size = min(1000, len(df))
sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)

print(f"✓ Analyse sur échantillon de {sample_size} codes...")

import_features = sample_df['source_code'].apply(detect_imports)
pattern_features = sample_df['source_code'].apply(detect_algo_patterns)

# Convertir en DataFrame
import_df = pd.DataFrame(import_features.tolist())
pattern_df = pd.DataFrame(pattern_features.tolist())

# Statistiques par tag
print(f"\n✓ Fréquence des imports par tag prioritaire (échantillon):")
for tag in PRIORITY_TAGS:
    mask = sample_df['tags'].apply(lambda tags: tag in tags)
    if mask.sum() > 0:
        tag_imports = import_df.loc[mask].mean()
        top_imports = tag_imports.nlargest(3)
        imports_str = ', '.join([f"{imp.replace('import_', '')}({val:.0%})" for imp, val in top_imports.items()])
        print(f"  {tag:20s}: {imports_str}")

print(f"\n✓ Fréquence des patterns par tag prioritaire (échantillon):")
for tag in PRIORITY_TAGS:
    mask = sample_df['tags'].apply(lambda tags: tag in tags)
    if mask.sum() > 0:
        tag_patterns = pattern_df.loc[mask].mean()
        top_patterns = tag_patterns[tag_patterns > 0.1].sort_values(ascending=False)
        if len(top_patterns) > 0:
            patterns_str = ', '.join([f"{pat.replace('has_', '')}({val:.0%})" for pat, val in top_patterns.head(3).items()])
            print(f"  {tag:20s}: {patterns_str}")

# ============================================================================
# 3. KEYWORD COVERAGE (LEXICAL RECALL)
# ============================================================================
print("\n[3/5] Analyse de couverture lexicale...")

# Définir des keywords par tag
TAG_KEYWORDS = {
    'math': ['number', 'sum', 'product', 'divide', 'multiply', 'calculate', 'formula', 'equation'],
    'graphs': ['graph', 'node', 'edge', 'vertex', 'path', 'connected', 'component', 'cycle'],
    'strings': ['string', 'substring', 'character', 'prefix', 'suffix', 'palindrome', 'pattern'],
    'number theory': ['prime', 'divisor', 'gcd', 'lcm', 'modulo', 'factor', 'coprime', 'remainder'],
    'trees': ['tree', 'root', 'parent', 'child', 'leaf', 'ancestor', 'descendant', 'subtree'],
    'geometry': ['point', 'line', 'angle', 'distance', 'coordinate', 'polygon', 'circle', 'area'],
    'games': ['game', 'player', 'win', 'lose', 'strategy', 'move', 'turn', 'optimal'],
    'probabilities': ['probability', 'expected', 'random', 'distribution', 'chance', 'likelihood']
}

def check_keyword_coverage(text, keywords):
    """Check if text contains any of the keywords"""
    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    return len(found) > 0, found

coverage_results = {}
for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    tag_df = df[mask]
    
    if len(tag_df) > 0:
        # Coverage in description
        desc_coverage = tag_df['prob_desc_description'].apply(
            lambda x: check_keyword_coverage(x, TAG_KEYWORDS[tag])
        )
        desc_has_kw = desc_coverage.apply(lambda x: x[0]).sum()
        
        # Coverage in code
        code_coverage = tag_df['source_code'].apply(
            lambda x: check_keyword_coverage(x, TAG_KEYWORDS[tag])
        )
        code_has_kw = code_coverage.apply(lambda x: x[0]).sum()
        
        # Coverage in either
        either_coverage = tag_df.apply(
            lambda row: check_keyword_coverage(row['prob_desc_description'], TAG_KEYWORDS[tag])[0] or
                       check_keyword_coverage(row['source_code'], TAG_KEYWORDS[tag])[0],
            axis=1
        ).sum()
        
        coverage_results[tag] = {
            'total': len(tag_df),
            'desc_coverage': desc_has_kw / len(tag_df),
            'code_coverage': code_has_kw / len(tag_df),
            'either_coverage': either_coverage / len(tag_df)
        }

print(f"✓ Couverture lexicale par tag:")
print(f"  {'Tag':20s} | {'Description':12s} | {'Code':12s} | {'Either':12s}")
print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
for tag, metrics in coverage_results.items():
    print(f"  {tag:20s} | {metrics['desc_coverage']:11.1%} | {metrics['code_coverage']:11.1%} | {metrics['either_coverage']:11.1%}")

# ============================================================================
# 4. OUTLIERS: HARD CASES
# ============================================================================
print("\n[4/5] Détection d'outliers (hard cases)...")

outliers_found = {}
for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    tag_df = df[mask]
    
    if len(tag_df) > 0:
        # Outliers de longueur (très courts)
        length_threshold = tag_df['length_chars'].quantile(0.05)
        very_short = tag_df[tag_df['length_chars'] < length_threshold]
        
        # Sans keywords lexicaux
        no_keywords = tag_df[tag_df['prob_desc_description'].apply(
            lambda x: not check_keyword_coverage(x, TAG_KEYWORDS[tag])[0]
        )]
        
        outliers_found[tag] = {
            'very_short_count': len(very_short),
            'no_keywords_count': len(no_keywords),
            'total': len(tag_df)
        }

print(f"✓ Outliers par tag:")
for tag, metrics in outliers_found.items():
    print(f"  {tag:20s}: {metrics['very_short_count']:3d} très courts, {metrics['no_keywords_count']:3d} sans keywords (sur {metrics['total']})")

# ============================================================================
# 5. SAUVEGARDE DES RÉSULTATS
# ============================================================================
print("\n[5/5] Sauvegarde des résultats...")

advanced_results = {
    "cooccurrence_lift": {
        f"{tag_a}+{tag_b}": {
            'lift': float(metrics['lift']),
            'pmi': float(metrics['pmi']),
            'count': int(metrics['count'])
        }
        for (tag_a, tag_b), metrics in sorted_by_lift[:20]
    },
    "keyword_coverage": {
        tag: {
            'desc_coverage': float(metrics['desc_coverage']),
            'code_coverage': float(metrics['code_coverage']),
            'either_coverage': float(metrics['either_coverage'])
        }
        for tag, metrics in coverage_results.items()
    },
    "outliers": {
        tag: {
            'very_short': int(metrics['very_short_count']),
            'no_keywords': int(metrics['no_keywords_count'])
        }
        for tag, metrics in outliers_found.items()
    }
}

with open('/app/docs/advanced_analysis_results.json', 'w') as f:
    json.dump(advanced_results, f, indent=2)

print(f"\n✅ Analyses avancées terminées")
print(f"  - Résultats: /app/docs/advanced_analysis_results.json")
print(f"\n📊 Insights clés:")
print(f"  - Lift max: {sorted_by_lift[0][1]['lift']:.2f} pour {sorted_by_lift[0][0]}")
print(f"  - Meilleure couverture lexicale: {max(coverage_results.items(), key=lambda x: x[1]['either_coverage'])[0]}")
print(f"  - Tag avec le plus d'outliers: {max(outliers_found.items(), key=lambda x: x[1]['no_keywords_count'])[0]}")

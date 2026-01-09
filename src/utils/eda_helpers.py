"""
Utility functions for Exploratory Data Analysis (EDA)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from collections import Counter
import ast


# Priority tags for the challenge
PRIORITY_TAGS = [
    'math', 'graphs', 'strings', 'number theory',
    'trees', 'geometry', 'games', 'probabilities'
]


def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load all JSON files from the dataset directory.
    
    Args:
        data_dir: Path to the directory containing JSON files
        
    Returns:
        DataFrame with all samples
    """
    import json
    from pathlib import Path
    
    data = []
    data_path = Path(data_dir)
    
    for file_path in data_path.glob('*.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples from {data_dir}")
    return df


def extract_tags_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tags column from string representation to actual list.
    
    Args:
        df: DataFrame with 'tags' column
        
    Returns:
        DataFrame with parsed tags
    """
    df = df.copy()
    
    # Parse tags if they're strings
    if df['tags'].dtype == 'object' and isinstance(df['tags'].iloc[0], str):
        df['tags'] = df['tags'].apply(ast.literal_eval)
    
    return df


def get_tag_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for all tags in the dataset.
    
    Args:
        df: DataFrame with 'tags' column (list of tags)
        
    Returns:
        DataFrame with tag statistics
    """
    # Flatten all tags
    all_tags = [tag for tags_list in df['tags'] for tag in tags_list]
    tag_counts = Counter(all_tags)
    
    stats = pd.DataFrame([
        {
            'tag': tag,
            'count': count,
            'frequency': count / len(df),
            'is_priority': tag in PRIORITY_TAGS
        }
        for tag, count in tag_counts.most_common()
    ])
    
    return stats


def get_priority_tag_coverage(df: pd.DataFrame) -> Dict:
    """
    Analyze coverage of priority tags in the dataset.
    
    Args:
        df: DataFrame with 'tags' column
        
    Returns:
        Dictionary with coverage statistics
    """
    # Count samples with at least one priority tag
    has_priority = df['tags'].apply(
        lambda tags: any(tag in PRIORITY_TAGS for tag in tags)
    )
    
    coverage = {
        'total_samples': len(df),
        'samples_with_priority_tag': has_priority.sum(),
        'coverage_percentage': (has_priority.sum() / len(df)) * 100,
        'samples_without_priority_tag': (~has_priority).sum()
    }
    
    return coverage


def analyze_tag_cooccurrence(df: pd.DataFrame, tags: List[str] = None) -> pd.DataFrame:
    """
    Compute co-occurrence matrix for tags.
    
    Args:
        df: DataFrame with 'tags' column
        tags: List of tags to analyze (default: PRIORITY_TAGS)
        
    Returns:
        Co-occurrence matrix as DataFrame
    """
    if tags is None:
        tags = PRIORITY_TAGS
    
    # Create binary matrix
    tag_matrix = pd.DataFrame(0, index=df.index, columns=tags)
    
    for idx, row_tags in enumerate(df['tags']):
        for tag in row_tags:
            if tag in tags:
                tag_matrix.loc[idx, tag] = 1
    
    # Compute co-occurrence
    cooccurrence = tag_matrix.T.dot(tag_matrix)
    
    return cooccurrence


def get_tag_examples(df: pd.DataFrame, tag: str, n: int = 3) -> pd.DataFrame:
    """
    Get sample examples for a specific tag.
    
    Args:
        df: DataFrame with 'tags' column
        tag: Tag to filter by
        n: Number of examples to return
        
    Returns:
        DataFrame with sample examples
    """
    mask = df['tags'].apply(lambda tags: tag in tags)
    samples = df[mask].head(n)
    
    return samples[['prob_desc_description', 'source_code', 'tags', 'difficulty']]


def compute_text_statistics(text_series: pd.Series) -> pd.DataFrame:
    """
    Compute statistics for text data.
    
    Args:
        text_series: Series of text data
        
    Returns:
        DataFrame with statistics
    """
    stats = pd.DataFrame({
        'length_chars': text_series.str.len(),
        'length_words': text_series.str.split().str.len(),
        'length_lines': text_series.str.count('\n') + 1
    })
    
    return stats


def extract_keywords_from_text(text: str, keywords: List[str]) -> List[str]:
    """
    Extract keywords present in text (case-insensitive).
    
    Args:
        text: Text to search
        keywords: List of keywords to find
        
    Returns:
        List of found keywords
    """
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]
    return found


def analyze_code_complexity(code: str) -> Dict:
    """
    Analyze Python code complexity using AST.
    
    Args:
        code: Python source code
        
    Returns:
        Dictionary with complexity metrics
    """
    try:
        tree = ast.parse(code)
        
        metrics = {
            'num_functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            'num_classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            'num_for_loops': len([n for n in ast.walk(tree) if isinstance(n, ast.For)]),
            'num_while_loops': len([n for n in ast.walk(tree) if isinstance(n, ast.While)]),
            'num_if_statements': len([n for n in ast.walk(tree) if isinstance(n, ast.If)]),
            'max_depth': _get_ast_depth(tree)
        }
    except SyntaxError:
        # If code can't be parsed, return zeros
        metrics = {
            'num_functions': 0,
            'num_classes': 0,
            'num_for_loops': 0,
            'num_while_loops': 0,
            'num_if_statements': 0,
            'max_depth': 0
        }
    
    return metrics


def _get_ast_depth(node, depth=0):
    """Helper function to compute AST depth."""
    if not isinstance(node, ast.AST):
        return depth
    
    max_child_depth = depth
    for child in ast.iter_child_nodes(node):
        child_depth = _get_ast_depth(child, depth + 1)
        max_child_depth = max(max_child_depth, child_depth)
    
    return max_child_depth


def plot_tag_distribution(tag_stats: pd.DataFrame, top_n: int = 20, highlight_priority: bool = True):
    """
    Plot tag distribution with optional highlighting of priority tags.
    
    Args:
        tag_stats: DataFrame from get_tag_statistics()
        top_n: Number of top tags to display
        highlight_priority: Whether to highlight priority tags
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = tag_stats.head(top_n)
    colors = ['#2ecc71' if is_priority else '#3498db' 
              for is_priority in data['is_priority']] if highlight_priority else '#3498db'
    
    ax.barh(data['tag'], data['count'], color=colors)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Tag', fontsize=12)
    ax.set_title(f'Top {top_n} Tags Distribution', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    if highlight_priority:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Priority Tags'),
            Patch(facecolor='#3498db', label='Other Tags')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return fig


def plot_cooccurrence_heatmap(cooccurrence_matrix: pd.DataFrame):
    """
    Plot co-occurrence heatmap for tags.
    
    Args:
        cooccurrence_matrix: Co-occurrence matrix from analyze_tag_cooccurrence()
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize by diagonal (self-occurrence)
    normalized = cooccurrence_matrix.copy()
    for i in range(len(normalized)):
        for j in range(len(normalized)):
            if i != j and normalized.iloc[i, i] > 0:
                normalized.iloc[i, j] = normalized.iloc[i, j] / normalized.iloc[i, i]
    
    sns.heatmap(normalized, annot=True, fmt='.2f', cmap='YlOrRd', 
                square=True, cbar_kws={'label': 'Co-occurrence Rate'}, ax=ax)
    ax.set_title('Tag Co-occurrence Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

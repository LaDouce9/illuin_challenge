"""
Preprocessing utilities for the Code Classification Challenge.

This module provides functions to clean, transform, and prepare the dataset
for machine learning pipelines. All functions are designed to be used in
sklearn pipelines with fit/transform patterns where applicable.

Author: Data Science Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import re


def clean_text_patterns(df: pd.DataFrame,
                        column: str,
                        patterns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Clean text patterns in a specified column.
    
    This function applies regex-based pattern corrections to fix common
    formatting issues in text data (e.g., "NoteIN" -> "Note: In").
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Name of the column to clean
    patterns : dict, optional
        Dictionary of {pattern: replacement}. If None, uses default patterns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column
        
    Examples
    --------
    >>> df = clean_text_patterns(df, 'prob_desc_notes')
    """
    df = df.copy()
    
    if patterns is None:
        patterns = {
            r'\bNoteIN\b': 'Note: In',
            r'\bNoteThe\b': 'Note: The',
            r'\bNoteA\b': 'Note: A',
            r'\bNoteAn\b': 'Note: An',
            r'\bNoteFor\b': 'Note: For',
            r'\bNoteIf\b': 'Note: If',
            r'\bNoteThat\b': 'Note: That'
        }
    
    for pattern, replacement in patterns.items():
        df[column] = df[column].apply(
            lambda x: re.sub(pattern, replacement, str(x)) if pd.notna(x) else x
        )
    
    return df


def handle_difficulty_invalid_values(df: pd.DataFrame,
                                      column: str = 'difficulty',
                                      invalid_value: int = -1) -> pd.DataFrame:
    """
    Replace invalid difficulty values with NaN.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str, default='difficulty'
        Name of the difficulty column
    invalid_value : int, default=-1
        Value to replace with NaN
        
    Returns
    -------
    pd.DataFrame
        DataFrame with invalid values replaced by NaN
    """
    df = df.copy()
    df[column] = df[column].replace(invalid_value, np.nan)
    return df


def impute_missing_values(df: pd.DataFrame,
                          columns: List[str],
                          strategy: str = 'median',
                          fill_values: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Impute missing values in specified columns.
    
    This function can work in two modes:
    - Training mode (fill_values=None): Compute statistics from data and return them
    - Inference mode (fill_values provided): Use pre-computed statistics
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list of str
        Columns to impute
    strategy : str, default='median'
        Imputation strategy ('median', 'mean', or 'mode')
    fill_values : dict, optional
        Pre-computed fill values {column: value}. If None, compute from data.
        
    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - DataFrame with imputed values
        - Dictionary of fill values used (to be saved for inference)
        
    Examples
    --------
    >>> # Training
    >>> df_train, fill_values = impute_missing_values(df_train, ['difficulty', 'time_limit_seconds'])
    >>> # Inference
    >>> df_test, _ = impute_missing_values(df_test, ['difficulty', 'time_limit_seconds'], 
    ...                                     fill_values=fill_values)
    """
    df = df.copy()
    
    if fill_values is None:
        fill_values = {}
        for col in columns:
            if col not in df.columns:
                continue
            
            if strategy == 'median':
                fill_values[col] = df[col].median()
            elif strategy == 'mean':
                fill_values[col] = df[col].mean()
            elif strategy == 'mode':
                fill_values[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].median()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    for col in columns:
        if col in df.columns and col in fill_values:
            df[col].fillna(fill_values[col], inplace=True)
    
    return df, fill_values


def create_priority_tags_column(df: pd.DataFrame,
                                 tags_column: str = 'tags',
                                 priority_tags: Optional[List[str]] = None,
                                 new_column: str = 'tags_priority') -> pd.DataFrame:
    """
    Create a new column containing only priority tags.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    tags_column : str, default='tags'
        Name of the original tags column
    priority_tags : list of str, optional
        List of priority tags to keep. If None, uses default priority tags.
    new_column : str, default='tags_priority'
        Name of the new column to create
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new tags_priority column
        
    Examples
    --------
    >>> df = create_priority_tags_column(df, tags_column='tags')
    >>> df['tags_priority'].head()
    0    [math, graphs]
    1    [strings]
    2    []
    """
    df = df.copy()
    
    if priority_tags is None:
        priority_tags = ['math', 'graphs', 'strings', 'number theory', 
                        'trees', 'geometry', 'games', 'probabilities']
    
    df[new_column] = df[tags_column].apply(
        lambda tags: [t for t in tags if t in priority_tags] if isinstance(tags, list) else []
    )
    
    return df


def encode_multilabel_target(df: pd.DataFrame,
                              tags_column: str = 'tags_priority',
                              priority_tags: Optional[List[str]] = None,
                              prefix: str = 'target_') -> pd.DataFrame:
    """
    Encode tags as binary multi-label columns.
    
    Creates one binary column per priority tag indicating presence/absence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    tags_column : str, default='tags_priority'
        Name of the tags column (should contain lists)
    priority_tags : list of str, optional
        List of priority tags. If None, uses default priority tags.
    prefix : str, default='target_'
        Prefix for the created binary columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with binary target columns added
        
    Examples
    --------
    >>> df = encode_multilabel_target(df, tags_column='tags_priority')
    >>> df[['target_math', 'target_graphs']].head()
       target_math  target_graphs
    0            1              1
    1            0              1
    2            1              0
    """
    df = df.copy()
    
    if priority_tags is None:
        priority_tags = ['math', 'graphs', 'strings', 'number theory', 
                        'trees', 'geometry', 'games', 'probabilities']
    
    for tag in priority_tags:
        col_name = f"{prefix}{tag.replace(' ', '_')}"
        df[col_name] = df[tags_column].apply(
            lambda tags: 1 if tag in tags else 0
        )
    
    return df


def extract_latex_binary_features(df: pd.DataFrame,
                                   latex_stats_df: pd.DataFrame,
                                   top_n: int = 30,
                                   min_frequency: int = 10,
                                   prefix: str = 'has_') -> pd.DataFrame:
    """
    Create binary features for top LaTeX symbols.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    latex_stats_df : pd.DataFrame
        Binary matrix (samples x symbols) from extract_all_latex_symbols()
    top_n : int, default=30
        Number of top symbols to keep as features
    min_frequency : int, default=10
        Minimum frequency threshold for a symbol to be included
    prefix : str, default='has_'
        Prefix for the created binary columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with binary LaTeX features added
        
    Examples
    --------
    >>> latex_stats = extract_all_latex_symbols(df, 'prob_desc_description')
    >>> df = extract_latex_binary_features(df, latex_stats, top_n=30)
    """
    df = df.copy()
    
    symbol_counts = latex_stats_df.sum().sort_values(ascending=False)
    
    valid_symbols = symbol_counts[
        (symbol_counts >= min_frequency) & 
        (symbol_counts.index.str.len() > 1)
    ].head(top_n)
    
    for symbol in valid_symbols.index:
        safe_name = symbol.replace('\\', '').replace('{', '').replace('}', '')
        col_name = f"{prefix}{safe_name}"
        df[col_name] = latex_stats_df[symbol].values
    
    return df


def create_text_length_features(df: pd.DataFrame,
                                 text_columns: List[str],
                                 compute_latex_ratio: bool = True,
                                 latex_density_columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Create text length features for specified columns.
    
    Features created for each text column:
    - {col}_char_length: Total number of characters
    - {col}_word_count: Number of words (split by whitespace)
    - {col}_numeric_ratio: Proportion of numeric characters (0-9) in the text
    - {col}_latex_ratio: LaTeX density (if compute_latex_ratio=True)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    text_columns : list of str
        Columns to compute length features for
    compute_latex_ratio : bool, default=True
        Whether to compute LaTeX ratio (requires latex_density columns)
    latex_density_columns : dict, optional
        Mapping {text_column: latex_density_column}
        
    Returns
    -------
    pd.DataFrame
        DataFrame with length features added
        
    Examples
    --------
    >>> df = create_text_length_features(df, ['prob_desc_description'])
    """
    df = df.copy()
    
    for col in text_columns:
        if col not in df.columns:
            continue
        
        df[f'{col}_char_length'] = df[col].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        
        df[f'{col}_word_count'] = df[col].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        df[f'{col}_numeric_ratio'] = df[col].apply(
            lambda x: sum(c.isdigit() for c in str(x)) / len(str(x)) if pd.notna(x) and len(str(x)) > 0 else 0.0
        )
        
        if compute_latex_ratio and latex_density_columns and col in latex_density_columns:
            latex_col = latex_density_columns[col]
            if latex_col in df.columns:
                df[f'{col}_latex_ratio'] = df[latex_col]
    
    return df


def remove_duplicate_rows(df: pd.DataFrame,
                          duplicate_column: str = 'duplicate_group') -> pd.DataFrame:
    """
    Remove duplicate rows, keeping only one representative per group.
    
    This function expects a 'duplicate_group' column created by
    detect_near_duplicates(). It keeps non-duplicates and the first
    sample of each duplicate group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with duplicate_group column
    duplicate_column : str, default='duplicate_group'
        Name of the column indicating duplicate groups
        
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    df = df.copy()
    
    if duplicate_column not in df.columns:
        return df
    
    non_duplicates = df[df[duplicate_column] == -1].copy()
    duplicates = df[df[duplicate_column] != -1].copy()
    
    duplicates_kept = duplicates.drop_duplicates(subset=[duplicate_column], keep='first')
    
    df_final = pd.concat([non_duplicates, duplicates_kept], ignore_index=True)
    
    return df_final


def validate_preprocessing(df: pd.DataFrame,
                           required_columns: Optional[List[str]] = None,
                           numeric_columns: Optional[List[str]] = None,
                           text_columns: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Validate that preprocessing was successful.
    
    Checks for missing values, invalid data types, and other issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe
    required_columns : list of str, optional
        Columns that must exist
    numeric_columns : list of str, optional
        Columns that should be numeric
    text_columns : list of str, optional
        Columns that should be text
        
    Returns
    -------
    dict
        Validation report with warnings and errors
        
    Examples
    --------
    >>> report = validate_preprocessing(df, required_columns=['tags_priority', 'unified_document'])
    >>> if report['errors']:
    ...     print("Preprocessing failed:", report['errors'])
    """
    report = {
        'errors': [],
        'warnings': [],
        'stats': {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_values': {}
        }
    }
    
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                report['errors'].append(f"Required column missing: {col}")
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if df[col].isna().any():
                    n_missing = df[col].isna().sum()
                    report['warnings'].append(f"Column {col} has {n_missing} missing values")
                    report['stats']['missing_values'][col] = n_missing
                
                if not pd.api.types.is_numeric_dtype(df[col]):
                    report['errors'].append(f"Column {col} should be numeric but is {df[col].dtype}")
    
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                if df[col].isna().any():
                    n_missing = df[col].isna().sum()
                    report['warnings'].append(f"Column {col} has {n_missing} missing values")
                    report['stats']['missing_values'][col] = n_missing
    
    return report


def print_preprocessing_step(step_name: str, n_samples_before: int, n_samples_after: int) -> None:
    """
    Print a formatted message for a preprocessing step.
    
    Parameters
    ----------
    step_name : str
        Name of the preprocessing step
    n_samples_before : int
        Number of samples before the step
    n_samples_after : int
        Number of samples after the step
    """
    change = n_samples_after - n_samples_before
    change_pct = (change / n_samples_before * 100) if n_samples_before > 0 else 0
    
    print(f"\n{'='*100}")
    print(f"STEP: {step_name}")
    print(f"{'='*100}")
    print(f"Samples before: {n_samples_before:,}")
    print(f"Samples after:  {n_samples_after:,}")
    
    if change != 0:
        sign = '+' if change > 0 else ''
        print(f"Change:         {sign}{change:,} ({sign}{change_pct:.2f}%)")
    
    print(f"{'='*100}")


def train_test_split_grouped(df: pd.DataFrame,
                              group_column: str = 'src_uid',
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets with grouping.
    
    Ensures that all samples with the same group_id are in the same split
    to avoid data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_column : str, default='src_uid'
        Column to group by (e.g., src_uid which is unique per solution)
    test_size : float, default=0.2
        Proportion of groups to include in test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - Train dataframe
        - Test dataframe
        
    Examples
    --------
    >>> df_train, df_test = train_test_split_grouped(df, group_column='code_uid')
    >>> print(f"Train: {len(df_train)}, Test: {len(df_test)}")
    """
    np.random.seed(random_state)
    
    # Get unique groups
    unique_groups = df[group_column].unique()
    n_groups = len(unique_groups)
    
    # Shuffle groups
    shuffled_groups = unique_groups.copy()
    np.random.shuffle(shuffled_groups)
    
    # Split groups
    n_test_groups = int(n_groups * test_size)
    test_groups = shuffled_groups[:n_test_groups]
    train_groups = shuffled_groups[n_test_groups:]
    
    # Create train and test sets
    df_train = df[df[group_column].isin(train_groups)].copy()
    df_test = df[df[group_column].isin(test_groups)].copy()
    
    return df_train, df_test


def get_preprocessing_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the preprocessed dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe
    """
    print(f"\n{'='*100}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*100}")
    print(f"Total samples:    {len(df):,}")
    print(f"Total features:   {len(df.columns):,}")
    print(f"\nMemory usage:     {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    missing = df.isna().sum()
    missing = missing[missing > 0]
    
    if len(missing) > 0:
        print(f"\nColumns with missing values:")
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count:,} ({pct:.2f}%)")
    else:
        print(f"\nNo missing values detected")
    
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if target_cols:
        print(f"\nTarget distribution (multi-label):")
        for col in sorted(target_cols):
            count = df[col].sum()
            pct = (count / len(df)) * 100
            tag_name = col.replace('target_', '').replace('_', ' ')
            print(f"  - {tag_name:20s}: {count:5,} ({pct:5.2f}%)")
    
    print(f"{'='*100}\n")


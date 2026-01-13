"""
Databook Generation Script
==========================

Generate a comprehensive Excel databook with all variable information.

Usage:
    python generate_databook.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. LOAD DATA
# ===========================

print("="*80)
print("DATABOOK GENERATION - Variable Dictionary")
print("="*80)

data_dir = Path('../data/processed')

print("\nLoading preprocessed datasets...")
df_train = pd.read_parquet(data_dir / 'train_preprocessed.parquet')
df_test = pd.read_parquet(data_dir / 'test_preprocessed.parquet')

print(f"Train set: {df_train.shape}")
print(f"Test set:  {df_test.shape}")

# Load imputation values
with open(data_dir / 'imputation_values.json', 'r') as f:
    imputation_values = json.load(f)

print(f"\nImputation values: {imputation_values}")

# ===========================
# 2. VARIABLE CLASSIFICATION
# ===========================

ORIGINAL_VARIABLES = [
    'prob_desc_time_limit', 'prob_desc_sample_outputs', 'src_uid',
    'prob_desc_notes', 'prob_desc_description', 'prob_desc_output_spec',
    'prob_desc_input_spec', 'prob_desc_output_to', 'prob_desc_input_from',
    'prob_desc_memory_limit', 'prob_desc_sample_inputs', 'prob_desc_created_at',
    'tags', 'lang_uid', 'prob_desc_description_link', 'file_name',
    'src', 'difficulty', 'code_uid', 'created_at', 'solution_uid'
]

all_columns = df_train.columns.tolist()
original_vars = [col for col in all_columns if col in ORIGINAL_VARIABLES]
created_vars = [col for col in all_columns if col not in ORIGINAL_VARIABLES]

print(f"\nOriginal variables: {len(original_vars)}")
print(f"Created variables:  {len(created_vars)}")
print(f"Total variables:    {len(all_columns)}")

# ===========================
# 3. CATEGORIZATION
# ===========================

def categorize_variable(col_name):
    """Categorize a variable based on its name pattern"""
    if col_name in ORIGINAL_VARIABLES:
        return 'Original'
    if '_translated' in col_name:
        return 'Translated'
    if col_name.startswith('has_') or 'latex' in col_name.lower():
        return 'LaTeX Feature'
    if any(x in col_name for x in ['char_length', 'word_count', 'latex_ratio', 'clean_']):
        return 'Text Feature'
    if col_name.startswith('target_'):
        return 'Target (Encoded)'
    if 'unified_document' in col_name:
        return 'Unified Document'
    if 'tags_priority' in col_name:
        return 'Target (Priority)'
    if col_name == 'time_limit_seconds':
        return 'Numeric Conversion'
    return 'Other Created'

var_categories = {col: categorize_variable(col) for col in all_columns}

print("\nVariables by category:")
print(pd.Series(var_categories).value_counts())

# ===========================
# 4. STATISTICS GENERATION
# ===========================

def get_variable_stats(df, col_name):
    """Get comprehensive statistics for a variable"""
    col = df[col_name]
    
    # Handle columns with unhashable types (like arrays)
    try:
        n_unique = col.nunique()
    except (TypeError, AttributeError):
        n_unique = 'N/A'
    
    stats = {
        'Variable': col_name,
        'Type': str(col.dtype),
        'N_Total': len(col),
        'N_Missing': col.isna().sum(),
        'Pct_Missing': f"{col.isna().sum() / len(col) * 100:.2f}%",
        'N_Unique': n_unique,
    }
    
    # Numeric statistics
    if pd.api.types.is_numeric_dtype(col):
        stats['Mean'] = f"{col.mean():.4f}" if not col.isna().all() else 'N/A'
        stats['Std'] = f"{col.std():.4f}" if not col.isna().all() else 'N/A'
        stats['Min'] = f"{col.min():.4f}" if not col.isna().all() else 'N/A'
        stats['Max'] = f"{col.max():.4f}" if not col.isna().all() else 'N/A'
        stats['Median'] = f"{col.median():.4f}" if not col.isna().all() else 'N/A'
    else:
        stats['Mean'] = 'N/A'
        stats['Std'] = 'N/A'
        stats['Min'] = 'N/A'
        stats['Max'] = 'N/A'
        stats['Median'] = 'N/A'
    
    # Most common value
    try:
        if isinstance(n_unique, int) and n_unique > 0 and not col.isna().all():
            most_common = col.value_counts().index[0]
            if isinstance(most_common, str) and len(most_common) > 50:
                most_common = most_common[:50] + '...'
            stats['Most_Common'] = str(most_common)
            stats['Most_Common_Count'] = int(col.value_counts().iloc[0])
        else:
            stats['Most_Common'] = 'N/A'
            stats['Most_Common_Count'] = 0
    except (TypeError, AttributeError, ValueError):
        stats['Most_Common'] = 'N/A'
        stats['Most_Common_Count'] = 0
    
    return stats

print("\nGenerating statistics...")
train_stats = [get_variable_stats(df_train, col) for col in all_columns]
df_train_stats = pd.DataFrame(train_stats)

# ===========================
# 5. ADD METADATA
# ===========================

df_train_stats['Category'] = df_train_stats['Variable'].map(var_categories)

def get_imputation_rule(col_name):
    """Get the imputation rule for a variable"""
    if col_name in imputation_values:
        return f"Median from TRAIN: {imputation_values[col_name]}"
    elif df_train[col_name].isna().sum() > 0:
        return "Contains missing values (no imputation)"
    else:
        return "No missing values"

df_train_stats['Imputation_Rule'] = df_train_stats['Variable'].apply(get_imputation_rule)

def get_variable_description(col_name):
    """Get a human-readable description"""
    descriptions = {
        'src_uid': 'Unique identifier for the solution',
        'code_uid': 'Unique identifier for the problem',
        'difficulty': 'Problem difficulty rating (1-3500)',
        'prob_desc_time_limit': 'Time limit for the problem (original string)',
        'prob_desc_description': 'Problem description (original)',
        'tags': 'Original tags (list)',
        'time_limit_seconds': 'Time limit converted to seconds (float)',
        'tags_priority': 'Filtered tags (priority tags only)',
        'unified_document': 'Concatenated text from all description fields',
        'unified_document_without_latex': 'Unified document with LaTeX replaced by [LATEX]',
        'clean_description': 'Description with LaTeX removed',
        'nb_latex_blocks': 'Number of LaTeX blocks in description',
        'nb_latex_symbols': 'Number of LaTeX symbols in description',
        'latex_density': 'Ratio of LaTeX characters to total characters',
    }
    
    if col_name in descriptions:
        return descriptions[col_name]
    elif col_name.startswith('target_'):
        tag_name = col_name.replace('target_', '').replace('_', ' ')
        return f"Binary target for tag: {tag_name}"
    elif col_name.startswith('has_'):
        symbol = col_name.replace('has_', '')
        return f"Binary: contains LaTeX symbol '{symbol}'"
    elif '_translated' in col_name:
        original = col_name.replace('_translated', '')
        return f"English translation of {original}"
    elif 'char_length' in col_name:
        return f"Character length of {col_name.replace('_char_length', '')}"
    elif 'word_count' in col_name:
        return f"Word count of {col_name.replace('_word_count', '')}"
    else:
        return "Variable created during preprocessing"

df_train_stats['Description'] = df_train_stats['Variable'].apply(get_variable_description)

# Reorder columns
column_order = [
    'Variable', 'Category', 'Description', 'Type', 'N_Total',
    'N_Missing', 'Pct_Missing', 'Imputation_Rule', 'N_Unique',
    'Mean', 'Std', 'Min', 'Median', 'Max',
    'Most_Common', 'Most_Common_Count'
]

df_train_stats = df_train_stats[column_order]
df_train_stats = df_train_stats.sort_values(['Category', 'Variable']).reset_index(drop=True)

# ===========================
# 6. SUMMARY STATISTICS
# ===========================

summary_stats = df_train_stats.groupby('Category').agg({
    'Variable': 'count',
    'N_Missing': 'sum'
}).rename(columns={'Variable': 'N_Variables', 'N_Missing': 'Total_Missing'})

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTotal variables:              {len(df_train_stats)}")
print(f"Original variables:           {(df_train_stats['Category'] == 'Original').sum()}")
print(f"Created variables:            {(df_train_stats['Category'] != 'Original').sum()}")
print(f"\nVariables with missing data:  {(df_train_stats['N_Missing'] > 0).sum()}")
print(f"Variables imputed:            {df_train_stats['Imputation_Rule'].str.contains('Median').sum()}")

# ===========================
# 7. EXPORT TO EXCEL
# ===========================

output_dir = Path('../docs')
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'databook_variables.xlsx'

print(f"\nExporting to Excel: {output_file}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # All variables
    df_train_stats.to_excel(writer, sheet_name='All_Variables', index=False)
    
    # Original variables
    df_original = df_train_stats[df_train_stats['Category'] == 'Original']
    df_original.to_excel(writer, sheet_name='Original_Variables', index=False)
    
    # Created variables
    df_created = df_train_stats[df_train_stats['Category'] != 'Original']
    df_created.to_excel(writer, sheet_name='Created_Variables', index=False)
    
    # Target variables
    df_target = df_train_stats[df_train_stats['Category'].str.contains('Target')]
    df_target.to_excel(writer, sheet_name='Target_Variables', index=False)
    
    # Variables with missing
    df_missing = df_train_stats[df_train_stats['N_Missing'] > 0]
    if len(df_missing) > 0:
        df_missing.to_excel(writer, sheet_name='Variables_With_Missing', index=False)
    
    # Summary
    summary_stats.to_excel(writer, sheet_name='Summary_By_Category')
    
    # Imputation rules
    imputation_df = pd.DataFrame([
        {'Variable': k, 'Imputation_Value': v, 'Source': 'Median from TRAIN set'}
        for k, v in imputation_values.items()
    ])
    imputation_df.to_excel(writer, sheet_name='Imputation_Rules', index=False)

print("\n" + "="*80)
print("DATABOOK GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\nFile location: {output_file.absolute()}")
print(f"\nSheets included:")
print("  1. All_Variables           - Complete databook")
print("  2. Original_Variables      - Original variables")
print("  3. Created_Variables       - Variables created during preprocessing")
print("  4. Target_Variables        - Target variables")
print("  5. Variables_With_Missing  - Variables with missing values")
print("  6. Summary_By_Category     - Summary statistics")
print("  7. Imputation_Rules        - Imputation rules applied")
print("\n" + "="*80)

# ===========================
# 8. DISPLAY KEY INFO
# ===========================

print("\nVARIABLES WITH MISSING DATA:")
missing_vars = df_train_stats[df_train_stats['N_Missing'] > 0][[
    'Variable', 'Category', 'N_Missing', 'Pct_Missing', 'Imputation_Rule'
]]

if len(missing_vars) > 0:
    print(missing_vars.to_string(index=False))
else:
    print("No variables with missing data (all imputed)")

print("\n" + "="*80)


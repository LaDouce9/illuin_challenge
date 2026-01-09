"""
Script to extract detailed EDA results for the report
"""
import sys
sys.path.append('/app')

import pandas as pd
import json

# Load the enriched dataset
df = pd.read_parquet('/app/data/processed/dataset_with_eda_features.parquet')

# Extract key statistics
results = {
    "dataset_info": {
        "total_samples": len(df),
        "total_columns": len(df.columns),
        "total_unique_tags": len(set([tag for tags in df['tags'] for tag in tags]))
    },
    "missing_values": {},
    "priority_tags": {},
    "tag_analysis": {},
    "language_distribution": {},
    "code_stats": {}
}

# Missing values
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        results["missing_values"][col] = {
            "count": int(missing_count),
            "percentage": float((missing_count / len(df)) * 100)
        }

# Priority tags
PRIORITY_TAGS = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
for tag in PRIORITY_TAGS:
    mask = df['tags'].apply(lambda tags: tag in tags)
    tag_df = df[mask]
    
    results["priority_tags"][tag] = {
        "count": int(len(tag_df)),
        "frequency": float(len(tag_df) / len(df)),
        "avg_difficulty": float(tag_df['difficulty'].mean()) if 'difficulty' in tag_df.columns and not tag_df['difficulty'].isna().all() else None,
        "avg_code_length": float(tag_df['source_code'].str.len().mean()),
        "avg_desc_length": float(tag_df['prob_desc_description'].str.len().mean())
    }

# Language distribution
lang_counts = df['lang'].value_counts()
for lang, count in lang_counts.items():
    results["language_distribution"][lang] = {
        "count": int(count),
        "percentage": float((count / len(df)) * 100)
    }

# Code statistics
results["code_stats"] = {
    "avg_length_chars": float(df['length_chars'].mean()),
    "median_length_chars": float(df['length_chars'].median()),
    "avg_length_lines": float(df['length_lines'].mean()),
    "median_length_lines": float(df['length_lines'].median())
}

# Number of tags per sample
df['num_tags'] = df['tags'].apply(len)
df['num_priority_tags'] = df['tags'].apply(lambda tags: sum(1 for t in tags if t in PRIORITY_TAGS))

results["tags_per_sample"] = {
    "avg_total_tags": float(df['num_tags'].mean()),
    "median_total_tags": float(df['num_tags'].median()),
    "avg_priority_tags": float(df['num_priority_tags'].mean()),
    "samples_with_priority": int((df['num_priority_tags'] > 0).sum()),
    "coverage_percentage": float(((df['num_priority_tags'] > 0).sum() / len(df)) * 100)
}

# Save results
with open('/app/docs/eda_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Results extracted and saved to /app/docs/eda_results.json")
print(json.dumps(results, indent=2))

# Train/Test Split Strategy

## 📊 Summary of Implementation

The preprocessing pipeline now includes a train/test split at the optimal point in the workflow, using a GroupSplit strategy to prevent data leakage.

---

## 🎯 Key Decisions

### 1. Split Ratio: 80% Train / 20% Test

**Decision**: No validation set, use **Cross-Validation** instead

**Rationale**:
- **80/20 split** provides enough data for both training and reliable evaluation
- **5-fold Cross-Validation** on train set for hyperparameter tuning
- Validation set only needed if extensive manual tuning is performed
- **Simpler workflow** with fewer splits to manage

**Alternative**: If you need a validation set for early stopping or extensive tuning:
- 70% train / 15% validation / 15% test
- Use the same GroupSplit strategy for all splits

---

### 2. GroupSplit on `src_uid` (CRITICAL)

**Decision**: Use **GroupSplit** on `src_uid` which is unique per solution

**Why this is straightforward**:
```
✅ SIMPLE SPLIT:
- Each `src_uid` is unique (one solution per src_uid)
- No risk of the same solution appearing in both train and test
- Standard train/test split with fixed random seed
- Ensures reproducibility and clean separation
```

**Implementation**:
```python
df_train, df_test = train_test_split_grouped(
    df,
    group_column='src_uid',       # Solution ID (unique)
    test_size=0.2,                # 20% of samples in test
    random_state=42               # Fixed for reproducibility
)
```

**Verification**:
- Train samples: ~3,978 (80%)
- Test samples: ~995 (20%)
- Overlap: 0 (verified in notebook)
- src_uid is unique, so each solution is in either train OR test

---

### 3. Label Stratification

**Decision**: **GroupSplit** takes priority over stratification

**Rationale**:
- **Preventing data leakage** (GroupSplit) is more important than perfect label balance
- Multi-label stratification is complex and may not be compatible with GroupSplit
- The dataset is large enough (4,973 samples) that random split provides reasonable label distribution

**Label Distribution Check** (after split):
```python
# Verify label distribution is similar in train and test
for tag in PRIORITY_TAGS:
    train_pct = df_train[f'target_{tag}'].mean()
    test_pct = df_test[f'target_{tag}'].mean()
    print(f"{tag}: Train {train_pct:.1%} | Test {test_pct:.1%}")
```

**Alternative**: If label imbalance is severe after split:
- Use `iterative-stratification` library (already in dependencies)
- Implement custom GroupSplit with stratification on majority tag
- May require more complex splitting logic

---

## 📍 Split Placement in Pipeline

**Position**: After deduplication, **before** any fitting operation

### Operations BEFORE split (no fitting):
1. ✅ Text pattern cleaning
2. ✅ Translation to English
3. ✅ Near-duplicate detection and removal
4. ✅ **→ TRAIN/TEST SPLIT ←**

### Operations AFTER split (some require fitting on train):
5. Time limit conversion (no fit)
6. Difficulty cleaning (no fit)
7. Text/LaTeX separation (no fit)
8. ⚠️ **LaTeX binary features** (fit on train: select top 30 symbols from TRAIN data only)
9. Text length features (no fit)
10. Unified document creation (no fit)
11. Target encoding (no fit, but classes from train)
12. ⚠️ **Missing value imputation** (fit on train: median from TRAIN data only)

---

## 🔄 Proper Fit/Transform Pattern

### Current Notebook Strategy

The notebook processes the **full dataset** for simplicity, with a note:
```
"For now, we continue processing the full dataset.
 The split will be used later for model training.
 Operations requiring 'fit' should be fitted on train only."
```

### Recommended Production Pattern

For proper ML pipeline:

```python
# 1. Split the data
df_train, df_test = train_test_split_grouped(df, ...)

# 2. Operations that need fitting - FIT ON TRAIN ONLY
# Example: LaTeX binary features
latex_stats_train = extract_all_latex_symbols(df_train, 'prob_desc_description_translated')
top_symbols = get_top_symbols(latex_stats_train, top_n=30)  # Get from train

# Apply to both train and test using the same symbols
df_train = create_binary_features(df_train, top_symbols)
df_test = create_binary_features(df_test, top_symbols)   # Same symbols!

# 3. Imputation - FIT ON TRAIN ONLY
median_difficulty = df_train['difficulty'].median()
median_time_limit = df_train['time_limit_seconds'].median()

df_train['difficulty'].fillna(median_difficulty, inplace=True)
df_test['difficulty'].fillna(median_difficulty, inplace=True)  # Same median!
```

---

## 🚀 Next Steps

### In Notebook 04 (Current)
- ✅ Split is defined and verified
- ✅ Full dataset is processed for feature engineering
- ⚠️ Note added: "Operations requiring 'fit' should use train data only"

### In Notebook 05 (Future: Modeling)
1. Load `df_train` and `df_test` from notebook 04
2. Properly fit all transformers on train:
   - TF-IDF: fit on `df_train['unified_document']`
   - Embeddings: fit on train (if using custom embeddings)
   - Feature selection: select on train
3. Transform both train and test using fitted transformers
4. Train models on train set
5. Evaluate on test set (only once, at the end!)

---

## 📊 Expected Split Statistics

```
Total samples:        4,973
Unique src_uid:       4,973 (each src_uid is unique)

After GroupSplit on src_uid:
  Train samples:      ~3,978 (80%)
  Test samples:       ~995 (20%)

src_uid overlap:      0 (verified)
```

---

## ⚠️ Common Pitfalls to Avoid

### ❌ BAD: Fit on full dataset, then split
```python
# DON'T DO THIS
df = impute_missing_values(df)  # Median computed on ALL data
df_train, df_test = split(df)   # Test has seen train data via median!
```

### ✅ GOOD: Split first, then fit on train
```python
# DO THIS
df_train, df_test = split(df)
median = df_train['col'].median()  # Median from TRAIN only
df_train['col'].fillna(median)
df_test['col'].fillna(median)      # Apply train median to test
```

### ❌ BAD: Random split (ignores groups)
```python
# DON'T DO THIS
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2)  # Leakage!
```

### ✅ GOOD: GroupSplit on src_uid
```python
# DO THIS
df_train, df_test = train_test_split_grouped(df, group_column='src_uid')
```

---

## 📚 References

- **GroupKFold**: [sklearn.model_selection.GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- **Multi-label Stratification**: [iterative-stratification](https://github.com/trent-b/iterative-stratification)
- **Data Leakage**: [Kaggle Learn - Data Leakage](https://www.kaggle.com/alexisbcook/data-leakage)

---

## 🎓 Summary

| Aspect | Decision | Reason |
|--------|----------|--------|
| **Split ratio** | 80/20 (train/test) | No validation set, use CV instead |
| **Strategy** | GroupSplit on `src_uid` | Each src_uid is unique, clean separation |
| **Stratification** | No | src_uid is unique, large dataset |
| **Placement** | After deduplication | Before any fitting operation |
| **Random seed** | 42 | Reproducibility |
| **Fit/Transform** | Fit on train only | Avoid leakage in imputation, feature selection |

**→ Bottom line**: Use GroupSplit on src_uid for clean separation, split before fitting, and verify zero overlap!


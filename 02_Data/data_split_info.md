# Data Split Information

## Dataset Overview

- **Total Records**: 284,807
- **Fraudulent Transactions (Class = 1)**: 492
- **Normal Transactions (Class = 0)**: 284,315
- **Fraud Ratio**: ~0.172%

---

## Splitting Strategy

- **Type**: Stratified Random Split
- **Purpose**: To preserve class ratio across train and test sets
- **Split Ratios**:
  - Train: 80%
  - Test: 20%

---

## Resulting Sizes

| Dataset | Total Samples | Fraud | Non-Fraud |
|---------|---------------|-------|-----------|
| Train   | 227,846       | 394   | 227,452   |
| Test    | 56,961        | 98    | 56,863    |

---

## Notes

- The split was performed using `train_test_split` from `sklearn.model_selection` with `stratify=y` to ensure class balance.
- `random_state=42` was used for reproducibility.
- Preprocessing (e.g., scaling) was fitted only on the training set and applied to the test set to avoid data leakage.
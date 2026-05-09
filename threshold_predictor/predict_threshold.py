import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import joblib


filename = "abs_diff.tsv"

def get_train_test_data(type='train'):
    # Shuffle and split: 80% train, 20% test.
    df = pd.read_csv(filename, sep='\t')
    errors = df['abs_diff'].values.astype(float)
    labels = df['labels'].values.astype(int)
    # Check that errors and labels have matching length.
    assert len(errors) == len(labels)
    # Shuffle data.
    indices = np.arange(len(errors))
    np.random.seed(42)
    np.random.shuffle(indices)
    errors = errors[indices]
    labels = labels[indices]
    # Split into train/test sets.
    split_idx = int(0.8 * len(errors))
    if type == 'train':
        return errors[:split_idx], labels[:split_idx]
    else:
        return errors[split_idx:], labels[split_idx:]

def evaluate_threshold(errors, y_true, T, metric='f1'):
    y_pred = (errors < T).astype(int)
    if metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'f1':
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        return f1
    else:
        raise ValueError("metric not supported")

def find_best_threshold_grid(errors: np.ndarray,
                             y_true: np.ndarray,
                             metric: str = 'f1',
                             n_candidates: int = 500) -> Tuple[float, float, np.ndarray, np.ndarray]:
    
    # Candidate thresholds use error percentiles (include 0 to max).
    candidates = np.unique(np.concatenate((
        [0.0],
        np.percentile(errors, np.linspace(0, 100, n_candidates)),
        [errors.max() + 1e-9]
    )))

    scores = [evaluate_threshold(errors, y_true, T, metric=metric) for T in candidates]
    assert len(scores) == len(candidates)

    best_idx = int(np.argmax(scores))
    assert isinstance(best_idx, int)
    T_star = candidates[best_idx]
    best_score = scores[best_idx]
    assert isinstance(T_star, float)
    assert isinstance(best_score, float)

    return T_star, best_score, candidates, scores

def dataset_level_features(errors: np.ndarray) -> Dict[str, float]:
    feats = {}
    feats['error_mean'] = float(np.mean(errors))
    feats['error_median'] = float(np.median(errors))
    feats['error_std'] = float(np.std(errors))
    feats['error_q25'] = float(np.percentile(errors, 25))
    feats['error_q75'] = float(np.percentile(errors, 75))
    feats['error_max'] = float(np.max(errors))
    feats['error_min'] = float(np.min(errors))

    return feats

def generate_regressor_examples(error_rt: np.ndarray,
                                y_true: np.ndarray,
                                n_bootstrap: int = 2000,
                                metric: str = 'f1') -> Tuple[pd.DataFrame, np.ndarray]:
    
    rng = np.random.default_rng(seed=42)
    rows = []
    targets = []
    n = len(error_rt)
    for _ in tqdm(range(n_bootstrap), desc='generating regressor examples'):
        idx = rng.choice(n, n, replace=True)
        errors_bs = error_rt[idx]
        y_bs = y_true[idx]
        T_star, _, _, _ = find_best_threshold_grid(errors_bs, y_bs, metric=metric, n_candidates=500)
        feats = dataset_level_features(errors_bs)
        rows.append(feats)
        targets.append(T_star)
    X_df = pd.DataFrame(rows)
    y_arr = np.array(targets, dtype=float)
    return X_df, y_arr

def train_threshold_regressor(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> RandomForestRegressor:
    """
    Fit a random-forest regressor: dataset-level features -> T*.
    """
    
    # Simple missing-value handling: fill with column medians.
    Xc = X.copy()
    Xc = Xc.fillna(Xc.median())

    rfr = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    rfr.fit(Xc.values, y)
    return rfr

def nested_cv_pipeline(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       error_col: str = 'error',
                       label_col: str = 'label',
                       metric: str = 'f1'):

    error_train = train_df[error_col].to_numpy()
    y_train = train_df[label_col].to_numpy().astype(int)

    # 1) Generate training samples for the threshold regressor.
    X_reg, y_reg = generate_regressor_examples(error_rt=error_train, y_true=y_train, metric=metric)

    # 2) Train the regressor.
    rfr = train_threshold_regressor(X_reg, y_reg, random_state=42)

    # Save trained regressor.
    joblib.dump(rfr, f'threshold_regressor.joblib')

    # 3) Evaluate on test.
    error_test = test_df[error_col].to_numpy()
    y_test = test_df[label_col].to_numpy().astype(int)

    feats_test = dataset_level_features(error_test)
    X_feat_test = pd.DataFrame([feats_test])
    X_feat_test = X_feat_test.fillna(X_feat_test.median())

    T_pred = float(rfr.predict(X_feat_test.values)[0])

    T_star_candidates = np.unique(np.concatenate(([0.0], np.percentile(error_test, np.linspace(0,100,501)), [error_test.max()+1e-9])))
    T_star_scores = [evaluate_threshold(error_test, y_test, T, metric='f1') for T in T_star_candidates]
    T_star_best_idx = np.argmax(T_star_scores)
    T_star_test = T_star_candidates[T_star_best_idx]

    # 4) Evaluate metrics under both thresholds.
    pred_acc = evaluate_threshold(error_test, y_test, T_pred, metric='accuracy')
    pred_f1 = evaluate_threshold(error_test, y_test, T_pred, metric='f1')
    opt_acc = evaluate_threshold(error_test, y_test, T_star_test, metric='accuracy')
    opt_f1 = evaluate_threshold(error_test, y_test, T_star_test, metric='f1')
    print(f"Predicted T_pred={T_pred:.4f}: accuracy={pred_acc:.4f}, f1={pred_f1:.4f}")
    print(f"Optimal T*_test={T_star_test:.4f}: accuracy={opt_acc:.4f}, f1={opt_f1:.4f}")

    # 5) Bootstrap comparison (CI for differences).
    acc_diffs = []
    f1_diffs = []
    rng = np.random.default_rng(seed=42)
    n_test = len(y_test)
    n_boot = 2000
    for _ in range(n_boot):
        idx = rng.choice(n_test, n_test, replace=True)
        e_boot = error_test[idx]
        yb = y_test[idx]
        val_acc = evaluate_threshold(e_boot, yb, T_star_test, metric='accuracy') - evaluate_threshold(e_boot, yb, T_pred, metric='accuracy')
        acc_diffs.append(val_acc)
        val_f1 = evaluate_threshold(e_boot, yb, T_star_test, metric='f1') - evaluate_threshold(e_boot, yb, T_pred, metric='f1')
        f1_diffs.append(val_f1)
    acc_diffs = np.array(acc_diffs)
    f1_diffs = np.array(f1_diffs)
    ci_low_acc, ci_high_acc = np.percentile(acc_diffs, [2.5, 97.5])
    p_approx_acc = np.mean(acc_diffs <= 0)  # proportion where T_star <= T_pred in terms of metric difference
    ci_low_f1, ci_high_f1 = np.percentile(f1_diffs, [2.5, 97.5])
    p_approx_f1 = np.mean(f1_diffs <= 0)  # proportion where T_star <= T_pred in terms of metric difference
    print(f"Bootstrap accuracy diff 95% CI: ({ci_low_acc:.4f}, {ci_high_acc:.4f}), approx p={p_approx_acc:.4f}")
    print(f"Bootstrap f1 diff 95% CI: ({ci_low_f1:.4f}, {ci_high_f1:.4f}), approx p={p_approx_f1:.4f}")


if __name__ == "__main__":
    # ----------------------- Training and evaluation -----------------------

    train_errors, train_labels = get_train_test_data(type='train')
    test_errors, test_labels = get_train_test_data(type='test')

    train_df = pd.DataFrame({
        'error': train_errors,
        'label': train_labels
    })
    test_df = pd.DataFrame({
        'error': test_errors,
        'label': test_labels
    })

    nested_cv_pipeline(train_df=train_df,
                       test_df=test_df,
                       error_col='error',
                       label_col='label',
                       metric='f1')

    print('\nFinished.')

    # ----------------------- Training and evaluation -----------------------

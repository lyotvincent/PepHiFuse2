import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


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


if __name__ == "__main__":
    # ----------------------- Grid search threshold -----------------------

    errors, y_true = get_train_test_data(type='test')
    # Grid search.
    candidates = np.unique(np.concatenate(([0.0], np.percentile(errors, np.linspace(0,100,501)), [errors.max()+1e-9])))
    scores = [evaluate_threshold(errors, y_true, T, metric='f1') for T in candidates]
    best_idx = np.argmax(scores)
    T_star = candidates[best_idx]
    best_score = scores[best_idx]
    acc_scores = [evaluate_threshold(errors, y_true, T, metric='accuracy') for T in candidates]
    corresponding_acc = acc_scores[best_idx]
    # Print candidates and scores separated by spaces.
    print("Candidates:", ' '.join([f"{c:.4f}" for c in candidates]))
    print("Scores:", ' '.join([f"{s:.4f}" for s in scores]))
    print(f"Best threshold T*: {T_star:.4f}, Best F1-score: {best_score:.4f}")
    print(f"Corresponding Accuracy: {corresponding_acc:.4f}")

    # ----------------------- Grid search threshold -----------------------

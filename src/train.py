"""
train.py
────────
Trains a Random Forest baseline classifier on the hand-crafted
EEG features extracted in features.py.

This baseline serves two purposes:
  1. Establishes a performance benchmark before we build the CNN
  2. Enables SHAP interpretability — showing which features
     drive each sleep stage classification

Pipeline:
  - Load features and labels
  - Split into train/validation/test sets (subject-aware)
  - Scale features
  - Train Random Forest with class weights
  - Save model and scaler
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ── constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR     = Path('results')
MODELS_DIR      = Path('models')
RANDOM_STATE    = 42
TEST_SIZE       = 0.2    # 20% held out for final test
VAL_SIZE        = 0.1    # 10% for validation during development

STAGE_NAMES     = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

STAGE_LABELS    = list(STAGE_NAMES.values())


# ── data loading ───────────────────────────────────────────────────────────────

def load_features():
    """
    Loads the feature matrix and labels saved by features.py.
    Validates shapes match before proceeding.
    """
    X_path = RESULTS_DIR / 'X_features.npy'
    y_path = RESULTS_DIR / 'y.npy'

    if not X_path.exists():
        raise FileNotFoundError('X_features.npy not found. Run features.py first.')
    if not y_path.exists():
        raise FileNotFoundError('y.npy not found. Run preprocess.py first.')

    X = np.load(X_path)
    y = np.load(y_path)

    if len(X) != len(y):
        raise ValueError(f'Shape mismatch: X has {len(X)} rows, y has {len(y)}')

    print(f'Loaded X_features: {X.shape}')
    print(f'Loaded y:          {y.shape}')
    return X, y


# ── data splitting ─────────────────────────────────────────────────────────────

def split_data(X, y):
    """
    Splits data into train, validation, and test sets using
    stratified sampling — ensuring each split has the same
    class proportions as the full dataset.

    This is important because our classes are imbalanced.
    Without stratification, the test set might contain very
    few N1 epochs by chance, making evaluation unreliable.
    """
    # first split: carve out test set
    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(splitter_test.split(X, y))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    X_test      = X[test_idx]
    y_test      = y[test_idx]

    # second split: carve validation out of remaining train data
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(splitter_val.split(X_train_val, y_train_val))

    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    X_val   = X_train_val[val_idx]
    y_val   = y_train_val[val_idx]

    print(f'\nData split:')
    print(f'  Train: {X_train.shape[0]} epochs')
    print(f'  Val:   {X_val.shape[0]} epochs')
    print(f'  Test:  {X_test.shape[0]} epochs')

    return X_train, y_train, X_val, y_val, X_test, y_test


# ── feature scaling ────────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
    """
    Standardises features to zero mean and unit variance.

    This is fitted on training data only — then applied to
    validation and test sets. Fitting on all data would leak
    information about the test set into training, which is
    a form of data leakage that inflates performance metrics.

    The scaler is saved so it can be applied consistently
    when the model is used on new data.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print('Scaler fitted and saved → models/scaler.pkl')

    return X_train, X_val, X_test, scaler


# ── model training ─────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier with class weights
    to handle the imbalanced sleep stage distribution.

    Random Forest works by building many decision trees on
    random subsets of the data, then combining their votes.
    It is robust, fast, and naturally handles non-linear
    relationships between features and labels.

    class_weight='balanced' automatically increases the
    penalty for misclassifying rare stages like N1 and REM.
    """
    print('\nTraining Random Forest...')

    model = RandomForestClassifier(
        n_estimators=500,       # number of trees
        max_depth=None,         # trees grow until leaves are pure
        min_samples_leaf=2,     # prevents overfitting on tiny groups
        class_weight='balanced',
        n_jobs=-1,              # use all CPU cores
        random_state=RANDOM_STATE,
        verbose=0
    )

    model.fit(X_train, y_train)
    print('Training complete.')
    return model


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, X_val, y_val, X_test, y_test):
    """
    Evaluates model on validation and test sets.
    Prints classification report and balanced accuracy.

    Balanced accuracy is more informative than raw accuracy
    for imbalanced datasets — it averages accuracy per class
    so rare stages are not drowned out by common ones.
    """
    print('\n── Validation set ──────────────────────────────')
    y_val_pred  = model.predict(X_val)
    val_bal_acc = balanced_accuracy_score(y_val, y_val_pred)
    print(f'Balanced accuracy: {val_bal_acc:.3f}')
    print(classification_report(
        y_val, y_val_pred,
        target_names=STAGE_LABELS,
        zero_division=0
    ))

    print('\n── Test set ────────────────────────────────────')
    y_test_pred  = model.predict(X_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    print(f'Balanced accuracy: {test_bal_acc:.3f}')
    print(classification_report(
        y_test, y_test_pred,
        target_names=STAGE_LABELS,
        zero_division=0
    ))

    return y_test_pred, test_bal_acc


def plot_confusion_matrix(y_test, y_test_pred):
    """
    Saves a normalised confusion matrix showing what percentage
    of each true stage was correctly classified or confused
    with another stage.

    Normalised by true label so each row sums to 1 —
    this makes rare classes (N1) readable alongside
    common ones (Wake, N2).
    """
    cm = confusion_matrix(y_test, y_test_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=STAGE_LABELS,
        yticklabels=STAGE_LABELS,
        ax=ax,
        vmin=0,
        vmax=1
    )
    ax.set_xlabel('Predicted stage', fontsize=12)
    ax.set_ylabel('True stage', fontsize=12)
    ax.set_title('Random Forest — normalised confusion matrix', fontsize=13)

    plt.tight_layout()
    path = RESULTS_DIR / 'confusion_matrix_rf.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'\nConfusion matrix saved → {path}')


def plot_feature_importance(model, feature_names):
    """
    Saves a bar chart of the top 15 most important features
    according to the Random Forest.

    Feature importance in a Random Forest measures how much
    each feature reduces impurity across all trees on average.
    High importance = the model relies heavily on this feature.

    This gives us a first look at which EEG characteristics
    matter most for sleep staging — before we apply SHAP.
    """
    importances     = model.feature_importances_
    indices         = np.argsort(importances)[::-1][:15]
    top_features    = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(len(top_features)),
        top_importances[::-1],
        color='steelblue',
        edgecolor='white'
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel('Feature importance', fontsize=12)
    ax.set_title('Top 15 features — Random Forest', fontsize=13)

    plt.tight_layout()
    path = RESULTS_DIR / 'feature_importance_rf.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Feature importance plot saved → {path}')


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from features import get_feature_names

    # step 1 — load data
    print('='*60)
    print('RANDOM FOREST BASELINE TRAINING')
    print('='*60)
    X, y = load_features()

    # step 2 — split
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # step 3 — scale
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # step 4 — train
    model = train_random_forest(X_train, y_train)

    # step 5 — evaluate
    y_test_pred, test_bal_acc = evaluate_model(
        model, X_val, y_val, X_test, y_test
    )

    # step 6 — plots
    feature_names = get_feature_names()
    plot_confusion_matrix(y_test, y_test_pred)
    plot_feature_importance(model, feature_names)

    # step 7 — save model
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / 'random_forest.pkl')
    print(f'Model saved → models/random_forest.pkl')

    print(f'\nBaseline complete. Test balanced accuracy: {test_bal_acc:.3f}')
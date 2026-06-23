"""
train.py
────────
Training pipeline for both the Random Forest baseline and the
1D CNN model for EEG sleep stage classification.

Run modes:
  python src/train.py --model rf    ← Random Forest baseline
  python src/train.py --model cnn   ← 1D CNN
  python src/train.py --model both  ← train both sequentially
"""

import argparse
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
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


# ── constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR     = Path('results')
MODELS_DIR      = Path('models')
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
VAL_SIZE        = 0.1

STAGE_NAMES     = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
STAGE_LABELS    = list(STAGE_NAMES.values())


# ── shared utilities ───────────────────────────────────────────────────────────

def split_data(X, y):
    """
    Stratified train/val/test split preserving class proportions.
    Fitted on features for RF, raw epochs for CNN — same indices used.
    """
    splitter_test = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(splitter_test.split(X, y))

    X_train_val = X[train_val_idx]
    y_train_val = y[train_val_idx]
    X_test      = X[test_idx]
    y_test      = y[test_idx]

    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(splitter_val.split(X_train_val, y_train_val))

    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    X_val   = X_train_val[val_idx]
    y_val   = y_train_val[val_idx]

    print(f'  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_confusion_matrix(y_test, y_pred, title, filename):
    """Saves a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=STAGE_LABELS, yticklabels=STAGE_LABELS,
        ax=ax, vmin=0, vmax=1
    )
    ax.set_xlabel('Predicted stage', fontsize=12)
    ax.set_ylabel('True stage', fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Confusion matrix saved → {path}')


from sklearn.metrics import cohen_kappa_score

def print_results(y_true, y_pred, split_name):
    """
    Prints balanced accuracy, Cohen's Kappa, and per-class
    classification report.

    Cohen's Kappa measures agreement between predictions and
    true labels, correcting for chance agreement. It is the
    standard metric in sleep staging literature:
      < 0.40  poor
      0.40–0.60  moderate
      0.60–0.80  good
      > 0.80  excellent
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa   = cohen_kappa_score(y_true, y_pred)

    print(f'\n── {split_name} ──────────────────────────────')
    print(f'Balanced accuracy : {bal_acc:.3f}')
    print(f'Cohen\'s Kappa     : {kappa:.3f}')
    print(classification_report(
        y_true, y_pred,
        target_names=STAGE_LABELS,
        zero_division=0
    ))
    return bal_acc, kappa

# ── random forest ──────────────────────────────────────────────────────────────

def train_random_forest():
    """Full Random Forest training pipeline."""
    print('\n' + '='*60)
    print('RANDOM FOREST BASELINE')
    print('='*60)

    # load features
    X = np.load(RESULTS_DIR / 'X_features.npy')
    y = np.load(RESULTS_DIR / 'y.npy')
    print(f'Features: {X.shape}')

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

    # train
    print('\nTraining Random Forest (500 trees)...')
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    # evaluate
    print_results(y_val,  rf.predict(X_val),  'Validation')
    bal_acc, kappa = print_results(y_test, rf.predict(X_test), 'Test')

    # plots
    plot_confusion_matrix(
        y_test, rf.predict(X_test),
        'Random Forest — normalised confusion matrix',
        'confusion_matrix_rf.png'
    )

    from features import get_feature_names
    feature_names   = get_feature_names()
    importances     = rf.feature_importances_
    indices         = np.argsort(importances)[::-1][:15]
    top_features    = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_features)), top_importances[::-1], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel('Feature importance', fontsize=12)
    ax.set_title('Top 15 features — Random Forest', fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importance_rf.png', dpi=150)
    plt.close()

    joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')
    print(f'\nRF baseline complete. Test balanced accuracy: {bal_acc:.3f}')
    return bal_acc


# ── 1d cnn ─────────────────────────────────────────────────────────────────────
def compute_class_weights(y_train):
    """
    Custom class weights that balance rare stage detection
    against Wake recall.

    Fully balanced weights overcorrect on small datasets —
    Wake recall drops to 0.30 because the model is penalised
    too heavily for missing rare stages. These manual weights
    reduce the Wake penalty while keeping N1 and REM prioritised.
    """
    class_weight_dict = {
    0: 0.75,  # Wake  — between 0.6 and 1.0
    1: 4.0,   # N1    — very rare, keep high
    2: 1.0,   # N2    — increase from 0.8 to help it compete
    3: 2.5,   # N3    — keep
    4: 2.5,   # REM   — keep
}

    print('\n  Class weights:')
    for cls, w in class_weight_dict.items():
        print(f'    {STAGE_NAMES[cls]:5s}: {w:.3f}')
    return class_weight_dict

def plot_training_history(history):
    """
    Saves a two-panel plot showing training and validation
    loss and accuracy across epochs.
    Useful for spotting overfitting — val loss rising while
    train loss keeps falling is the warning sign.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # loss
    axes[0].plot(history.history['loss'],     label='Train loss')
    axes[0].plot(history.history['val_loss'], label='Val loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and validation loss')
    axes[0].legend()

    # accuracy
    axes[1].plot(history.history['accuracy'],     label='Train accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and validation accuracy')
    axes[1].legend()

    plt.tight_layout()
    path = RESULTS_DIR / 'cnn_training_history.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Training history saved → {path}')


def train_cnn():
    """Full 1D CNN training pipeline."""
    print('\n' + '='*60)
    print('1D CNN TRAINING')
    print('='*60)

    # load raw epochs — CNN works on raw signal not features
    X = np.load(RESULTS_DIR / 'X.npy')
    y = np.load(RESULTS_DIR / 'y.npy')
    print(f'Raw epochs: {X.shape}')

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # normalise signal amplitude across training set
    # we compute mean and std from training data only
    mean    = X_train.mean()
    std     = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_val   = (X_val   - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)

    # save normalisation stats for inference later
    MODELS_DIR.mkdir(exist_ok=True)
    np.save(MODELS_DIR / 'cnn_norm_stats.npy', np.array([mean, std]))
    print(f'  Signal normalised — mean: {mean:.6f}, std: {std:.6f}')

    # class weights
    class_weights = compute_class_weights(y_train)

    # build and compile model
    from model import build_1d_cnn, compile_model, get_callbacks
    cnn = build_1d_cnn(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        n_classes=5
    )
    cnn = compile_model(cnn)
    print(f'\n  Parameters: {cnn.count_params():,}')

    # train
    print('\nTraining CNN...')
    print('(EarlyStopping will halt if val_loss stops improving)\n')

    history = cnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=get_callbacks(MODELS_DIR),
        verbose=1
    )

    # evaluate
    y_val_pred  = np.argmax(cnn.predict(X_val,  verbose=0), axis=1)
    y_test_pred = np.argmax(cnn.predict(X_test, verbose=0), axis=1)

    print_results(y_val,  y_val_pred,  'Validation')
    bal_acc, kappa = print_results(y_test, y_test_pred, 'Test')

    # plots
    plot_training_history(history)
    plot_confusion_matrix(
        y_test, y_test_pred,
        '1D CNN — normalised confusion matrix',
        'confusion_matrix_cnn.png'
    )

    print(f'\nCNN training complete.')
    print(f'Test balanced accuracy : {bal_acc:.3f}')
    print(f'Test Cohen\'s Kappa     : {kappa:.3f}')
    return bal_acc


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='Train sleep staging models')
    parser.add_argument(
        '--model',
        choices=['rf', 'cnn', 'both'],
        default='both',
        help='Which model to train (default: both)'
    )
    args = parser.parse_args()

    if args.model == 'rf':
        train_random_forest()
    elif args.model == 'cnn':
        train_cnn()
    else:
        rf_acc  = train_random_forest()
        cnn_acc = train_cnn()
        print('\n' + '='*60)
        print('RESULTS SUMMARY')
        print('='*60)
        print(f'Random Forest  — balanced accuracy: {rf_acc:.3f}')
        print(f'1D CNN         — balanced accuracy: {cnn_acc:.3f}')
        improvement = (cnn_acc - rf_acc) / rf_acc * 100
        print(f'CNN improvement over RF: {improvement:+.1f}%')
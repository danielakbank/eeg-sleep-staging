"""
evaluate.py
───────────
Generates interpretability outputs for the trained models:

  1. SHAP analysis on the Random Forest
     — shows which EEG features drive each sleep stage prediction
     — produces summary plot and per-stage bar charts

  2. Grad-CAM on the 1D CNN
     — highlights which time segments within a 30-second epoch
       the model focuses on when making its prediction
     — overlaid directly on the raw EEG signal

These outputs serve two purposes:
  - Clinical validation: confirming the model uses neuroscientifically
    meaningful patterns (delta waves for N3, spindle-frequency for N2)
  - Research transparency: interpretable AI is a core requirement
    for clinical deployment, directly relevant to the EPIC project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import tensorflow as tf
from pathlib import Path


# ── constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR     = Path('results')
MODELS_DIR      = Path('models')
SFREQ           = 100
EPOCH_DURATION  = 30
SAMPLES         = SFREQ * EPOCH_DURATION   # 3000

STAGE_NAMES     = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
STAGE_COLOURS   = {
    0: 'steelblue',
    1: 'mediumpurple',
    2: 'seagreen',
    3: 'coral',
    4: 'goldenrod'
}


# ── shap analysis ──────────────────────────────────────────────────────────────

def run_shap_analysis(n_background=200, n_explain=100):
    """
    Runs SHAP TreeExplainer on the Random Forest model.

    SHAP (SHapley Additive exPlanations) assigns each feature
    a value representing its contribution to a specific prediction.
    Positive SHAP = pushed prediction toward this class.
    Negative SHAP = pushed prediction away from this class.

    We use a background sample to estimate expected feature values,
    then explain a test sample to see how features shift predictions
    away from that baseline.
    """
    print('\n' + '='*60)
    print('SHAP ANALYSIS — Random Forest')
    print('='*60)

    rf_path = MODELS_DIR / 'random_forest.pkl'
    if not rf_path.exists():
        print('  Random Forest not found. Run train.py --model rf first.')
        return

    rf          = joblib.load(rf_path)
    scaler      = joblib.load(MODELS_DIR / 'scaler.pkl')
    X_features  = np.load(RESULTS_DIR / 'X_features.npy')
    y           = np.load(RESULTS_DIR / 'y.npy')

    X_scaled = scaler.transform(X_features)

    rng             = np.random.RandomState(42)
    bg_idx          = rng.choice(len(X_scaled), n_background, replace=False)
    exp_idx         = rng.choice(len(X_scaled), n_explain,    replace=False)
    X_background    = X_scaled[bg_idx]
    X_explain       = X_scaled[exp_idx]

    print(f'  Background samples : {n_background}')
    print(f'  Explanation samples: {n_explain}')
    print('  Computing SHAP values (this takes a minute)...')

    explainer   = shap.TreeExplainer(rf, X_background)
    shap_values = explainer.shap_values(X_explain)

    from features import get_feature_names
    feature_names = get_feature_names()

    # ── plot 1: per-stage bar charts ──────────────────────────────────────────
    print('  Generating SHAP summary plots...')
    fig, axes = plt.subplots(1, 5, figsize=(22, 7))

    for class_idx, stage_name in STAGE_NAMES.items():
        ax          = axes[class_idx]
        mean_shap   = np.abs(shap_values[class_idx]).mean(axis=0)
        sorted_idx  = np.argsort(mean_shap)[::-1][:10]
        top_names   = [feature_names[i] for i in sorted_idx]
        top_vals    = mean_shap[sorted_idx]

        ax.barh(
            range(len(top_names)),
            top_vals[::-1],
            color=STAGE_COLOURS[class_idx],
            edgecolor='white',
            alpha=0.85
        )
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(
            [n.replace('_', '\n') for n in top_names[::-1]],
            fontsize=7
        )
        ax.set_title(stage_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean |SHAP|', fontsize=9)

    plt.suptitle(
        'SHAP feature importance per sleep stage — Random Forest',
        fontsize=13
    )
    plt.tight_layout()
    path = RESULTS_DIR / 'shap_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  SHAP summary saved → {path}')

    # ── plot 2: heatmap across all classes ────────────────────────────────────
    shap_matrix = np.array([
        np.abs(shap_values[c]).mean(axis=0)
        for c in range(5)
    ]).T   # shape: (n_features, n_classes)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        shap_matrix,
        xticklabels=list(STAGE_NAMES.values()),
        yticklabels=[n.replace('_', ' ') for n in feature_names],
        cmap='YlOrRd',
        ax=ax,
        annot=False,
        linewidths=0.3
    )
    ax.set_title(
        'Feature importance across all sleep stages (mean |SHAP|)',
        fontsize=13
    )
    ax.set_xlabel('Sleep stage', fontsize=11)
    ax.set_ylabel('EEG feature', fontsize=11)
    plt.tight_layout()
    path = RESULTS_DIR / 'shap_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  SHAP heatmap saved → {path}')

    return shap_values, feature_names


# ── grad-cam analysis ──────────────────────────────────────────────────────────

def compute_gradcam_1d(model, epoch, class_idx, conv_layer_name='conv3'):
    """
    Computes Grad-CAM activation map for a 1D CNN.

    Grad-CAM works by:
    1. Running a forward pass to get the prediction
    2. Computing gradients of the target class score with respect
       to the last conv layer's output feature maps
    3. Averaging gradients across all filters
    4. Weighting the feature maps by those averaged gradients
    5. Applying ReLU — only positive contributions matter

    The result is a 1D map showing which time segments drove
    the prediction, upsampled back to 3000 timepoints.
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.output
        ]
    )

    epoch_batch = np.expand_dims(epoch, axis=0)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(epoch_batch)
        target_score = predictions[:, class_idx]

    grads   = tape.gradient(target_score, conv_output)
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam     = tf.reduce_sum(conv_output[0] * weights, axis=-1).numpy()

    # relu and normalise
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    # upsample to original signal length
    cam_upsampled = np.interp(
        np.linspace(0, len(cam) - 1, SAMPLES),
        np.arange(len(cam)),
        cam
    )

    return cam_upsampled, predictions.numpy()[0]


def run_gradcam_analysis():
    """
    Generates Grad-CAM visualisations for one correctly classified
    example epoch per sleep stage. The activation map is overlaid
    as a shaded background on the raw EEG signal — brighter shading
    means higher model attention at that time segment.
    """
    print('\n' + '='*60)
    print('GRAD-CAM ANALYSIS — 1D CNN')
    print('='*60)

    cnn_path = MODELS_DIR / 'best_cnn.keras'
    if not cnn_path.exists():
        print('  CNN model not found. Run train.py --model cnn first.')
        return

    cnn     = tf.keras.models.load_model(cnn_path)
    X       = np.load(RESULTS_DIR / 'X.npy')
    y       = np.load(RESULTS_DIR / 'y.npy')

    norm_stats  = np.load(MODELS_DIR / 'cnn_norm_stats.npy')
    mean, std   = norm_stats[0], norm_stats[1]
    X_norm      = (X - mean) / (std + 1e-8)

    times   = np.arange(SAMPLES) / SFREQ
    fig, axes = plt.subplots(5, 1, figsize=(16, 18))

    for stage_idx, stage_name in STAGE_NAMES.items():
        ax          = axes[stage_idx]
        stage_mask  = y == stage_idx
        stage_X     = X_norm[stage_mask]
        stage_X_raw = X[stage_mask]

        if len(stage_X) == 0:
            ax.set_title(f'{stage_name} — no examples found')
            ax.axis('off')
            continue

        # find a correctly predicted example
        found = False
        for i in range(min(50, len(stage_X))):
            epoch       = stage_X[i]
            cam, probs  = compute_gradcam_1d(cnn, epoch, stage_idx)
            pred_class  = np.argmax(probs)
            if pred_class == stage_idx:
                raw_epoch = stage_X_raw[i]
                found = True
                break

        if not found:
            epoch       = stage_X[0]
            cam, probs  = compute_gradcam_1d(cnn, epoch, stage_idx)
            raw_epoch   = stage_X_raw[0]
            pred_class  = np.argmax(probs)

        # convert to microvolts
        signal_uv = raw_epoch[:, 0] * 1e6

        # set y range based on signal amplitude before plotting
        margin  = np.abs(signal_uv).max() * 1.3
        y_min   = -margin
        y_max   =  margin

        # overlay grad-cam shading first (behind the signal)
        # split into segments so alpha varies per time point
        for t in range(len(times) - 1):
            ax.fill_between(
                times[t:t+2],
                y_min,
                y_max,
                alpha=float(cam[t]) * 0.45,
                color=STAGE_COLOURS[stage_idx],
                linewidth=0,
                zorder=1
            )

        # plot raw EEG signal on top
        ax.plot(
            times,
            signal_uv,
            color='#1a1a2e',
            linewidth=0.7,
            alpha=0.9,
            label='Fpz-Cz',
            zorder=2
        )

        ax.set_ylim(y_min, y_max)

        pred_name   = STAGE_NAMES[pred_class]
        confidence  = probs[pred_class] * 100
        ax.set_title(
            f'{stage_name} — predicted: {pred_name} '
            f'({confidence:.1f}% confidence)',
            fontsize=11
        )
        ax.set_ylabel('Amplitude (µV)', fontsize=9)

        # add a colour patch to the legend for Grad-CAM
        from matplotlib.patches import Patch
        legend_elements = [
            plt.Line2D([0], [0], color='#1a1a2e', linewidth=1, label='Fpz-Cz'),
            Patch(facecolor=STAGE_COLOURS[stage_idx], alpha=0.45,
                  label='Grad-CAM activation')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (seconds)', fontsize=10)
    plt.suptitle(
        'Grad-CAM: time segments driving CNN predictions\n'
        'Shaded regions = high model attention',
        fontsize=13
    )
    plt.tight_layout()
    path = RESULTS_DIR / 'gradcam_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Grad-CAM plot saved → {path}')


# ── results summary ────────────────────────────────────────────────────────────

def generate_results_summary():
    """
    Prints a concise summary of all results suitable for
    copying into the README.
    """
    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)
    print("""
Model comparison (13 subjects, Sleep-EDF Cassette subset):

  Random Forest (hand-crafted features)
  ├── Balanced accuracy : 0.312
  ├── Cohen's Kappa     : -0.024
  └── Best stage        : N3 (F1 0.40)

  1D CNN (raw EEG signal)
  ├── Balanced accuracy : 0.718
  ├── Cohen's Kappa     : 0.430  (moderate agreement)
  ├── Best stages       : N3 (0.94 recall), REM (0.93 recall)
  └── Hardest stage     : Wake (0.28 recall)

Key findings:
  1. CNN outperforms RF by 130% on 13 subjects
     (RF degrades with more subjects — spectral features
      do not generalise across individuals)
  2. Theta power is the most informative feature for Wake,
     N1 and N2 — alpha power absence defines N3
  3. REM classification improved most from RF to CNN
     confirming CNNs learn morphological waveform patterns
     beyond what frequency content alone captures
  4. Wake recall remains low — temporal context via LSTM
     is the natural next step

Limitations:
  - Adult healthy subjects (Sleep-EDF) vs target (paediatric NDC)
  - 13 subjects limits cross-subject generalisation
  - No temporal sequencing between epochs (LSTM extension)
    """)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True)

    run_shap_analysis()
    run_gradcam_analysis()
    generate_results_summary()
    
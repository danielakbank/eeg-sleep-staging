"""
features.py
───────────
Extracts hand-crafted features from preprocessed EEG epochs.
Each epoch is transformed from a raw signal (3000, 2) into a
compact feature vector that captures the key characteristics
of each sleep stage.

Features extracted per channel:
  - Delta band power      (0.5–4 Hz)  — dominant in N3
  - Theta band power      (4–8 Hz)    — dominant in N1
  - Alpha band power      (8–13 Hz)   — dominant in relaxed wake
  - Beta band power       (13–30 Hz)  — dominant in active wake
  - Gamma band power      (30–40 Hz)  — high frequency activity
  - Relative band powers  (each band / total power)
  - Spectral edge freq    (95% of power below this frequency)
  - Signal variance
  - Hjorth mobility       (signal complexity measure)
  - Hjorth complexity     (signal complexity measure)

Total features: 14 per channel × 2 channels = 28 features per epoch
"""

import numpy as np
from scipy import signal
from pathlib import Path


# ── frequency band definitions ─────────────────────────────────────────────────

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 40.0),
}

SFREQ           = 100     # Hz
EPOCH_DURATION  = 30      # seconds
RESULTS_DIR     = Path('results')


# ── feature extraction functions ───────────────────────────────────────────────

def compute_band_powers(channel_data, sfreq=SFREQ):
    """
    Computes absolute and relative power in each frequency band
    using Welch's power spectral density method.

    Welch's method splits the signal into overlapping windows,
    computes the FFT of each, then averages them. This gives a
    more stable estimate than a single FFT on the whole epoch.

    Returns a dict of absolute powers and relative powers per band.
    """
    # compute power spectral density using Welch's method
    freqs, psd = signal.welch(
        channel_data,
        fs=sfreq,
        nperseg=sfreq * 4,    # 4-second windows
        noverlap=sfreq * 2,   # 50% overlap
    )

    total_power = np.trapz(psd, freqs)

    powers      = {}
    rel_powers  = {}

    for band_name, (low, high) in BANDS.items():
        # find frequency indices within this band
        band_mask       = (freqs >= low) & (freqs < high)
        band_power      = np.trapz(psd[band_mask], freqs[band_mask])
        powers[band_name]       = band_power
        rel_powers[band_name]   = band_power / (total_power + 1e-10)

    return powers, rel_powers


def compute_spectral_edge(channel_data, sfreq=SFREQ, edge=0.95):
    """
    Finds the frequency below which 95% of the total signal power sits.

    In deep sleep (N3) this will be low — most power is in slow delta waves.
    In Wake it will be higher — power is spread across faster frequencies.
    """
    freqs, psd  = signal.welch(channel_data, fs=sfreq, nperseg=sfreq * 4)
    cumulative  = np.cumsum(psd)
    cumulative /= cumulative[-1]   # normalise to 0–1
    edge_idx    = np.searchsorted(cumulative, edge)
    return freqs[min(edge_idx, len(freqs) - 1)]


def compute_hjorth_parameters(channel_data):
    """
    Computes Hjorth mobility and complexity — two measures of
    signal complexity used widely in EEG analysis.

    Mobility: measures how fast the signal is changing.
              High in Wake, lower in deep sleep.

    Complexity: measures how irregular those changes are.
                High in Wake, lower in rhythmic sleep stages.
    """
    diff1 = np.diff(channel_data)
    diff2 = np.diff(diff1)

    var0  = np.var(channel_data)
    var1  = np.var(diff1)
    var2  = np.var(diff2)

    mobility    = np.sqrt(var1 / (var0 + 1e-10))
    complexity  = np.sqrt(var2 / (var1 + 1e-10)) / (mobility + 1e-10)

    return mobility, complexity


def extract_features_single_epoch(epoch):
    """
    Extracts all features from a single epoch.

    Input:  epoch of shape (3000, 2)
    Output: feature vector of shape (40,)
    """
    features = []

    for ch_idx in range(epoch.shape[1]):
        channel_data = epoch[:, ch_idx]

        # band powers
        powers, rel_powers = compute_band_powers(channel_data)

        for band in BANDS:
            features.append(powers[band])
            features.append(rel_powers[band])

        # spectral edge frequency
        features.append(compute_spectral_edge(channel_data))

        # signal variance
        features.append(np.var(channel_data))

        # hjorth parameters
        mobility, complexity = compute_hjorth_parameters(channel_data)
        features.append(mobility)
        features.append(complexity)

    return np.array(features)


def extract_features_all_epochs(X):
    """
    Extracts features from all epochs.

    Input:  X of shape (n_epochs, 3000, 2)
    Output: feature matrix of shape (n_epochs, 40)
    """
    print(f'Extracting features from {len(X)} epochs...')
    print('This may take a few minutes.\n')

    n_epochs    = len(X)
    features    = []

    for i, epoch in enumerate(X):
        features.append(extract_features_single_epoch(epoch))

        # progress indicator every 500 epochs
        if (i + 1) % 500 == 0 or (i + 1) == n_epochs:
            pct = (i + 1) / n_epochs * 100
            print(f'  {i+1}/{n_epochs} epochs processed ({pct:.1f}%)')

    features = np.array(features)
    print(f'\nFeature matrix shape: {features.shape}')
    return features


def get_feature_names():
    """
    Returns the name of every feature in the same order
    they appear in the feature vector. Useful for SHAP
    interpretability plots later.
    """
    names = []
    channel_names = ['Fpz-Cz', 'Pz-Oz']

    for ch in channel_names:
        for band in BANDS:
            names.append(f'{ch}_{band}_power')
            names.append(f'{ch}_{band}_rel_power')
        names.append(f'{ch}_spectral_edge')
        names.append(f'{ch}_variance')
        names.append(f'{ch}_hjorth_mobility')
        names.append(f'{ch}_hjorth_complexity')

    return names


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # load preprocessed data
    X_path = RESULTS_DIR / 'X.npy'
    y_path = RESULTS_DIR / 'y.npy'

    if not X_path.exists():
        raise FileNotFoundError(
            'X.npy not found. Run preprocess.py first.'
        )

    print('Loading preprocessed data...')
    X = np.load(X_path)
    y = np.load(y_path)
    print(f'Loaded X: {X.shape}, y: {y.shape}')

    # extract features
    X_features = extract_features_all_epochs(X)

    # print feature names so we know what we extracted
    feature_names = get_feature_names()
    print(f'\nFeatures extracted ({len(feature_names)} total):')
    for i, name in enumerate(feature_names):
        print(f'  [{i:2d}] {name}')

    # save feature matrix
    np.save(RESULTS_DIR / 'X_features.npy', X_features)
    print(f'\nSaved → results/X_features.npy')
    print(f'Shape: {X_features.shape}  (n_epochs, n_features)')
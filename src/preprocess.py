"""
preprocess.py
─────────────
Loads raw Sleep-EDF recordings, trims to the actual sleep window,
applies bandpass filtering, rejects artefact epochs, and saves clean
numpy arrays ready for feature extraction and model training.

Designed to work for any subject in the Sleep-EDF dataset by detecting
sleep start and end times automatically from the hypnogram annotations.
"""

import os
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ── constants ──────────────────────────────────────────────────────────────────

DATA_DIR        = Path('data/physionet-sleep-data')
RESULTS_DIR     = Path('results')
EEG_CHANNELS    = ['EEG Fpz-Cz', 'EEG Pz-Oz']
SFREQ           = 100          # sampling rate in Hz
EPOCH_DURATION  = 30           # seconds per epoch (AASM standard)
SAMPLES_PER_EPOCH = SFREQ * EPOCH_DURATION   # 3000 samples
BANDPASS_LOW    = 0.5          # Hz — removes slow drift
BANDPASS_HIGH   = 40.0         # Hz — removes high-frequency noise
ARTEFACT_THRESHOLD = 150e-6    # 150 µV — flags epochs with movement noise
SLEEP_BUFFER    = 30 * 60      # 30 minutes either side of sleep window (seconds)

# map old Rechtschaffen & Kales labels → modern 5-class system
STAGE_MAP = {
    'Sleep stage W':  0,   # Wake
    'Sleep stage 1':  1,   # N1
    'Sleep stage 2':  2,   # N2
    'Sleep stage 3':  3,   # N3  (old stage 3)
    'Sleep stage 4':  3,   # N3  (old stage 4 — merged with stage 3)
    'Sleep stage R':  4,   # REM
    'Sleep stage ?': -1,   # Unknown — will be discarded
}


# ── helper functions ───────────────────────────────────────────────────────────

def find_sleep_window(annotations):
    """
    Detects the start and end of the actual sleep period from hypnogram
    annotations. Works for any subject regardless of when they fell asleep.

    Returns (sleep_start_seconds, sleep_end_seconds).
    """
    sleep_stages = [
        ann for ann in annotations
        if ann['description'] not in ('Sleep stage W', 'Sleep stage ?')
    ]

    if not sleep_stages:
        raise ValueError('No sleep stages found in annotations.')

    sleep_start = sleep_stages[0]['onset']
    last_stage  = sleep_stages[-1]
    sleep_end   = last_stage['onset'] + last_stage['duration']

    # add buffer either side so we don't cut off transitions
    sleep_start = max(0, sleep_start - SLEEP_BUFFER)
    sleep_end   = sleep_end + SLEEP_BUFFER

    print(f'  Sleep window detected: '
          f'{sleep_start/3600:.2f}hrs → {sleep_end/3600:.2f}hrs')

    return sleep_start, sleep_end


def load_and_trim(psg_path, hypnogram_path):
    """
    Loads raw EEG, keeps only the two EEG channels, applies annotations,
    and trims the recording to the sleep window only.
    """
    print(f'  Loading {psg_path.name}...')
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # keep only the two EEG channels — drop EOG, EMG, temperature etc.
    raw.pick_channels(EEG_CHANNELS)

    # attach hypnogram annotations to the raw object
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, verbose=False)

    # find sleep window and crop
    sleep_start, sleep_end = find_sleep_window(annotations)
    sleep_end = min(sleep_end, raw.times[-1])   # don't exceed recording length
    raw.crop(tmin=sleep_start, tmax=sleep_end)

    print(f'  Trimmed duration: {(sleep_end - sleep_start)/3600:.2f} hours')
    return raw, annotations, sleep_start


def bandpass_filter(raw):
    """
    Applies a bandpass filter to keep only 0.5–40 Hz.
    This removes slow electrical drift (below 0.5 Hz) and
    high-frequency noise like muscle artefacts (above 40 Hz).
    """
    print('  Applying bandpass filter (0.5–40 Hz)...')
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method='fir',
        verbose=False
    )
    return raw


def extract_epochs(raw, annotations, recording_start_seconds):
    """
    Slices the filtered signal into 30-second epochs and assigns
    a sleep stage label to each one.

    Returns:
        epochs_data  — numpy array of shape (n_epochs, 3000, 2)
        labels       — numpy array of shape (n_epochs,)
        valid_mask   — which epochs passed artefact rejection
    """
    print('  Extracting epochs...')

    sfreq        = raw.info['sfreq']
    data, times  = raw[EEG_CHANNELS]          # shape: (2, n_samples)
    data         = data.T                      # shape: (n_samples, 2)

    epochs_data  = []
    labels       = []

    for ann in annotations:
        stage       = ann['description']
        label       = STAGE_MAP.get(stage, -1)

        # skip unknown stages
        if label == -1:
            continue

        # find where this annotation falls within the trimmed recording
        onset_in_trimmed = ann['onset'] - recording_start_seconds
        start_sample     = int(onset_in_trimmed * sfreq)

        # annotation duration may cover multiple 30-second epochs
        n_epochs_in_ann = int(ann['duration'] / EPOCH_DURATION)

        # annotations shorter than 30 seconds are sub-epoch — skip them
        if n_epochs_in_ann == 0:
            continue

        for i in range(n_epochs_in_ann):
            epoch_start = start_sample + i * SAMPLES_PER_EPOCH
            epoch_end   = epoch_start + SAMPLES_PER_EPOCH

            # skip if epoch goes beyond the recording
            if epoch_end > len(data):
                continue

            epoch = data[epoch_start:epoch_end]   # shape: (3000, 2)

            # skip if epoch is the wrong length
            if epoch.shape[0] != SAMPLES_PER_EPOCH:
                continue

            epochs_data.append(epoch)
            labels.append(label)

    epochs_data = np.array(epochs_data)   # (n_epochs, 3000, 2)
    labels      = np.array(labels)

    print(f'  Extracted {len(epochs_data)} epochs before artefact rejection')
    return epochs_data, labels


def reject_artefacts(epochs_data, labels):
    """
    Removes epochs where the signal amplitude exceeds 150 µV.
    These are almost always movement artefacts or electrode noise
    rather than genuine brain activity.
    """
    print('  Rejecting artefact epochs...')

    max_amplitude = np.abs(epochs_data).max(axis=(1, 2))
    clean_mask    = max_amplitude < ARTEFACT_THRESHOLD

    n_rejected = np.sum(~clean_mask)
    print(f'  Rejected {n_rejected} artefact epochs '
          f'({n_rejected/len(epochs_data)*100:.1f}%)')

    return epochs_data[clean_mask], labels[clean_mask]


def print_class_distribution(labels):
    """
    Prints how many epochs exist per sleep stage after preprocessing.
    Useful for understanding class imbalance before training.
    """
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    print('\n  Class distribution after preprocessing:')
    for stage_id, stage_name in stage_names.items():
        count = np.sum(labels == stage_id)
        pct   = count / len(labels) * 100
        bar   = '█' * int(pct / 2)
        print(f'    {stage_name:5s} (class {stage_id}): '
              f'{count:4d} epochs ({pct:5.1f}%) {bar}')


def save_subject_plot(epochs_data, labels, subject_id):
    """
    Saves a quick visual summary of cleaned epochs per stage.
    One example epoch waveform per sleep stage.
    """
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    RESULTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    times     = np.arange(SAMPLES_PER_EPOCH) / SFREQ

    for stage_id, stage_name in stage_names.items():
        ax      = axes[stage_id]
        mask    = labels == stage_id
        indices = np.where(mask)[0]

        if len(indices) == 0:
            ax.set_title(f'{stage_name} — no epochs found')
            ax.axis('off')
            continue

        # plot first available clean epoch for this stage
        epoch = epochs_data[indices[0]]
        ax.plot(times, epoch[:, 0] * 1e6,
                linewidth=0.6, color='steelblue', label='Fpz-Cz')
        ax.plot(times, epoch[:, 1] * 1e6,
                linewidth=0.6, color='coral', alpha=0.7, label='Pz-Oz')
        ax.set_title(f'{stage_name} — example clean epoch')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (seconds)')
    plt.suptitle(f'Subject {subject_id} — one clean epoch per sleep stage',
                 fontsize=13)
    plt.tight_layout()

    plot_path = RESULTS_DIR / f'subject_{subject_id}_epochs.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {plot_path}')


# ── main pipeline ──────────────────────────────────────────────────────────────

def preprocess_subject(psg_path, hypnogram_path, subject_id):
    """
    Runs the full preprocessing pipeline for a single subject.
    Returns clean epochs array and labels array.
    """
    print(f'\n{"="*60}')
    print(f'Processing subject {subject_id}')
    print(f'{"="*60}')

    # step 1 — load and trim to sleep window
    raw, annotations, recording_start = load_and_trim(psg_path, hypnogram_path)

    # step 2 — bandpass filter
    raw = bandpass_filter(raw)

    # step 3 — extract epochs and labels
    epochs_data, labels = extract_epochs(raw, annotations, recording_start)

    # step 4 — reject artefact epochs
    epochs_data, labels = reject_artefacts(epochs_data, labels)

    # step 5 — print class distribution
    print_class_distribution(labels)

    # step 6 — save example epoch plot
    save_subject_plot(epochs_data, labels, subject_id)

    print(f'\n  Final dataset: {epochs_data.shape} epochs, '
          f'{labels.shape[0]} labels')

    return epochs_data, labels


def preprocess_all_subjects():
    """
    Finds all PSG/Hypnogram pairs in the data directory and
    preprocesses each one. Saves combined arrays to results/.
    """
    # find all PSG files
    psg_files = sorted(DATA_DIR.glob('*PSG.edf'))

    if not psg_files:
        raise FileNotFoundError(f'No PSG files found in {DATA_DIR}')

    all_epochs = []
    all_labels = []

    for psg_path in psg_files:
        # derive hypnogram filename from PSG filename
        subject_id      = psg_path.name[:6]
        hypnogram_name  = psg_path.name.replace('PSG.edf', '') 
        
        # find matching hypnogram
        hypnogram_files = list(DATA_DIR.glob(f'{subject_id}*Hypnogram.edf'))

        if not hypnogram_files:
            print(f'  No hypnogram found for {psg_path.name} — skipping')
            continue

        hypnogram_path = hypnogram_files[0]

        epochs, labels = preprocess_subject(psg_path, hypnogram_path, subject_id)

        all_epochs.append(epochs)
        all_labels.append(labels)

    # combine all subjects into single arrays
    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f'\n{"="*60}')
    print(f'ALL SUBJECTS COMBINED')
    print(f'{"="*60}')
    print(f'X shape: {X.shape}   (n_epochs, timepoints, channels)')
    print(f'y shape: {y.shape}   (n_epochs,)')
    print_class_distribution(y)

    # save to results folder
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / 'X.npy', X)
    np.save(RESULTS_DIR / 'y.npy', y)
    print(f'\nSaved → results/X.npy')
    print(f'Saved → results/y.npy')

    return X, y


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    X, y = preprocess_all_subjects()
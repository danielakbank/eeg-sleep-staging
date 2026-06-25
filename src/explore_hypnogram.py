import mne

hypnogram = 'data/physionet-sleep-data/SC4001EC-Hypnogram.edf'
annotations = mne.read_annotations(hypnogram)

print('=== All annotations ===')
for i, ann in enumerate(annotations):
    onset_hrs = ann['onset'] / 3600
    duration_mins = ann['duration'] / 60
    stage = ann['description']
    print(f'[{i:3d}] onset: {onset_hrs:.2f}hrs | duration: {duration_mins:.0f}min | stage: {stage}')
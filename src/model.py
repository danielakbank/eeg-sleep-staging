"""
model.py
────────
Defines the 1D CNN architecture for EEG sleep stage classification.

Architecture:
  - Two Conv1D blocks learn local waveform patterns from raw EEG
  - GlobalAveragePooling collapses the time dimension
  - Dense head with dropout produces 5-class softmax output

The model takes raw EEG epochs as input — no hand-crafted features
needed. The convolutional filters learn their own feature detectors
directly from the signal, including sleep spindles, K-complexes,
and delta wave patterns that band power features miss.

Input shape:  (batch_size, 3000, 2)
Output shape: (batch_size, 5)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from pathlib import Path


# ── architecture ───────────────────────────────────────────────────────────────

def build_1d_cnn(input_shape=(3000, 2), n_classes=5, l2_reg=1e-4):
    """
    Builds and returns the 1D CNN model.

    Args:
        input_shape: tuple of (timepoints, channels)
                     default (3000, 2) = 30s at 100Hz, 2 EEG channels
        n_classes:   number of sleep stages to classify (5)
        l2_reg:      L2 regularisation strength — penalises large
                     weights to reduce overfitting on small datasets

    Returns:
        compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape, name='eeg_input')

    # ── block 1 ───────────────────────────────────────────────────────────────
    # kernel_size=50 spans 0.5 seconds at 100Hz
    # this is long enough to detect a full sleep spindle (0.5–2s)
    # 64 filters = 64 different pattern detectors running in parallel
    x = layers.Conv1D(
        filters=64,
        kernel_size=50,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1'
    )(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling1D(pool_size=4, name='pool1')(x)
    # output shape: (750, 64)

    # ── block 2 ───────────────────────────────────────────────────────────────
    # kernel_size=25 spans 0.25 seconds — detects finer patterns
    # after pooling, the signal is compressed so shorter kernel is appropriate
    # 128 filters — deeper layers learn more complex combinations
    x = layers.Conv1D(
        filters=128,
        kernel_size=25,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling1D(pool_size=4, name='pool2')(x)
    # output shape: (187, 128)

    # ── block 3 ───────────────────────────────────────────────────────────────
    # third block captures higher-level abstractions
    # smaller kernel — working with already-compressed representations
    x = layers.Conv1D(
        filters=256,
        kernel_size=10,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv3'
    )(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)

    

    # ── global average pooling ────────────────────────────────────────────────
    # collapses (187, 256) → (256,)
    # averages across all time positions — much better than Flatten
    # for preventing overfitting on small datasets
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # ── classifier head ───────────────────────────────────────────────────────
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense1'
    )(x)
    x = layers.Dropout(0.5, name='dropout1')(x)

    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense2'
    )(x)
    x = layers.Dropout(0.3, name='dropout2')(x)

    outputs = layers.Dense(
        n_classes,
        activation='softmax',
        name='output'
    )(x)

    model = models.Model(inputs, outputs, name='eeg_sleep_cnn')
    return model


def compile_model(model):
    """
    Compiles the model with Adam optimiser and categorical
    cross-entropy loss.

    sparse_categorical_crossentropy accepts integer labels directly
    — no need to one-hot encode y before training.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(models_dir):
    """
    Returns training callbacks:

    EarlyStopping — stops training when val_loss stops improving.
    Prevents overfitting by not training longer than necessary.
    restore_best_weights=True means we keep the best checkpoint
    automatically, not the final epoch weights.

    ReduceLROnPlateau — halves the learning rate when val_loss
    plateaus for 3 epochs. Helps the model fine-tune once it
    has found a good region of the loss landscape.

    ModelCheckpoint — saves the best model weights to disk
    so we can reload them later for evaluation.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / 'best_cnn.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
    ]
    return callbacks


def print_model_summary(model):
    """
    Prints a readable summary of the model architecture
    including output shapes and parameter counts per layer.
    """
    print('\n' + '='*60)
    print('MODEL ARCHITECTURE')
    print('='*60)
    model.summary()
    total_params = model.count_params()
    print(f'\nTotal parameters: {total_params:,}')
    print('='*60 + '\n')


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = build_1d_cnn()
    model = compile_model(model)
    print_model_summary(model)
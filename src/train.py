import os
import numpy as np
import tensorflow as tf
from src.preprocess import get_data_generators
from src.model import build_model

DATA_DIR = "chest_xray/chest_xray"
MODEL_SAVE_PATH = "models/best_model.h5"
EPOCHS = 30

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def train():
    train_gen, val_gen, _ = get_data_generators(DATA_DIR)

    # Class imbalance fix
    total = 5216
    normal = 1341
    pneumonia = 3875
    class_weight = {
        0: total / (2 * normal),
        1: total / (2 * pneumonia)
    }
    print("Class weights:", class_weight)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-7
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks
    )

    return history, model

if __name__ == "__main__":
    history, model = train()
    print("\n✅ Training Complete!")
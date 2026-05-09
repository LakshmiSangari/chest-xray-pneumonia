import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import tensorflow as tf
from src.preprocess import get_data_generators

DATA_DIR = "chest_xray/chest_xray"
MODEL_PATH = "models/best_model.h5"

def evaluate():
    _, _, test_gen = get_data_generators(DATA_DIR)

    model = tf.keras.models.load_model(MODEL_PATH)

    # Predictions
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred,
          target_names=['NORMAL', 'PNEUMONIA']))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.show()
    print("✅ Confusion matrix saved to results/")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/roc_curve.png')
    plt.show()
    print(f"✅ ROC-AUC Score: {auc:.4f}")

if __name__ == "__main__":
    evaluate()

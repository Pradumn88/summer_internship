import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import load_datasets, MODEL_DIR, MODEL_FILE  # uses your existing paths

def main():
    # load data
    _, _, test_ds, _, _, _ = load_datasets()

    # load model WITHOUT old compile state
    model = keras.models.load_model(MODEL_FILE, compile=False)

    # compile fresh with correct metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", curve="PR"),
        ],
    )

    print("\n🧪 Evaluating on test set (clean eval)...")
    results = model.evaluate(test_ds)
    names = model.metrics_names  # ['loss','accuracy','precision','recall','auc']

    print("\n📊 Clean Test Metrics:")
    for n, v in zip(names, results):
        if n in ["accuracy", "precision", "recall", "auc"]:
            print(f"  {n:<10}: {v:.2%}")
        else:
            print(f"  {n:<10}: {v:.4f}")

    # save separate clean report
    lines = ["Clean Evaluation Report", f"Date: {datetime.datetime.now()}"]
    for n, v in zip(names, results):
        if n in ["accuracy", "precision", "recall", "auc"]:
            lines.append(f"{n}: {v:.4%}")
        else:
            lines.append(f"{n}: {v:.4f}")

    report_path = os.path.join(MODEL_DIR, "evaluation_report_clean.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n📝 Clean report saved to {report_path}")

if __name__ == "__main__":
    main()

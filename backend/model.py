import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt                     # <-- ADDED
from sklearn.metrics import confusion_matrix         # <-- ADDED
from sklearn.metrics import ConfusionMatrixDisplay   # <-- ADDED

from tensorflow.keras import layers, models, applications, callbacks
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ================= PATHS =================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chest_xray")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 8

BASE_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_FILE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(MODEL_DIR, "checkpoint_model.h5")
FINAL_MODEL_FILE = os.path.join(MODEL_DIR, "multiclass_xray_model.keras")

# ================= DATA LOADING =================

def load_datasets():
    print("\n📂 Loading datasets...")

    for split in ["train", "valid", "test"]:
        path = os.path.join(BASE_DIR, split)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split: {path}")
        print(f"✔ {split}: {path}")

    train_gen_raw = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.mobilenet_v2.preprocess_input,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.mobilenet_v2.preprocess_input
    )

    train_ds = train_gen_raw.flow_from_directory(
        os.path.join(BASE_DIR, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_ds = val_test_gen.flow_from_directory(
        os.path.join(BASE_DIR, "valid"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    test_ds = val_test_gen.flow_from_directory(
        os.path.join(BASE_DIR, "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_ds.class_indices.keys())
    class_counts = np.bincount(train_ds.classes)
    total_samples = int(class_counts.sum())

    print(f"🏷️ Classes: {class_names}")
    print(f"📊 Train counts: {dict(zip(class_names, class_counts))}")
    print(f"🧮 Total train samples: {total_samples}")

    return train_ds, val_ds, test_ds, class_names, class_counts, total_samples


# ================= MODEL CREATION =================

def create_model(num_classes):
    print("\n🧠 Building MobileNetV2-based model...")

    data_aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )

    base_model = applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SIZE + (3,),
        pooling="avg",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMAGE_SIZE + (3,))
    x = data_aug(inputs)
    x = base_model(x, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="multiclass_xray_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", curve="PR"),
        ],
    )

    model.summary()
    return model


# ================= TRAIN & FINE-TUNE =================

def train_and_fine_tune(model, train_ds, val_ds, class_counts):
    print("\n🎯 Training with class-balanced loss...")

    labels = train_ds.classes
    classes = np.unique(labels)

    class_weights_arr = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
    print("⚖️ Class weights:", class_weights)

    log_dir = os.path.join(MODEL_DIR, "logs_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_cb = callbacks.ModelCheckpoint(
        CHECKPOINT_FILE,
        save_best_only=True,
        monitor="val_auc",
        mode="max",
        verbose=1
    )

    cb_list = [
        checkpoint_cb,
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        callbacks.EarlyStopping(monitor="val_auc", patience=7, restore_best_weights=True),
        callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv")),
    ]

    # ---- STAGE 1 ----
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=cb_list,
        class_weight=class_weights,
    )

    # ---- STAGE 2 Fine-tuning ----
    print("\n🔧 Fine-tuning base model...")
    base_model = model.get_layer("mobilenetv2_1.00_224")
    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc", curve="PR"),
        ],
    )

    history_fine = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=cb_list,
        class_weight=class_weights,
    )

    return history, history_fine


# ================= EVALUATION =================

def evaluate_model(model, test_ds):
    print("\n🧪 Evaluating on test set...")

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

    results = model.evaluate(test_ds)

    report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Evaluation Results\n")
        for name, val in zip(model.metrics_names, results):
            f.write(f"{name}: {val}\n")

    return results


# =================== ADDED: TRAINING CURVES PLOT ===================

def plot_curves(history1, history2):                       # <-- ADDED
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Training & Validation Curves")
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
    plt.close()


# =================== ADDED: CONFUSION MATRIX ===================

def create_confusion_matrix(model, test_ds, class_names):        # <-- ADDED
    y_true = test_ds.classes
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()


# ================= MAIN =================

if __name__ == "__main__":
    print("\n🚀 START TRAINING\n")

    train_ds, val_ds, test_ds, class_names, class_counts, total = load_datasets()
    model = create_model(len(class_names))

    h1, h2 = train_and_fine_tune(model, train_ds, val_ds, class_counts)

    evaluate_model(model, test_ds)

    # SAVE FINAL MODEL
    model.save(FINAL_MODEL_FILE)
    print(f"\n💾 Final model saved to: {FINAL_MODEL_FILE}")

    # SAVE CLASS NAMES
    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
        for c in class_names:
            f.write(c + "\n")

    # NEW: TRAINING CURVES
    plot_curves(h1, h2)                              # <-- ADDED
    print("📊 Saved: training_curves.png")

    # NEW: CONFUSION MATRIX
    create_confusion_matrix(model, test_ds, class_names)    # <-- ADDED
    print("📊 Saved: confusion_matrix.png")

    print("\n✅ Training complete!")

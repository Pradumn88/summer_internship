import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chest_xray")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 8

BASE_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_FILE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "multiclass_xray_model.keras")

# ---------- DATA LOADING ----------

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

# ---------- MIXUP GENERATOR ----------



# ---------- MODEL CREATION ----------

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

# ---------- TRAIN & FINE-TUNE ----------

def train_and_fine_tune(model, train_ds, val_ds, class_counts):
    print("\n🎯 Training with class-balanced loss (no MixUp in final run)...")

    # ----- class weights for imbalance -----
    labels = train_ds.classes
    classes = np.unique(labels)
    class_weights_arr = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
    print(f"⚖️ Class weights used in training: {class_weights}")

    log_dir = os.path.join(
        BASE_FILE_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(log_dir, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            patience=7, monitor="val_auc", mode="max", restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_FILE,
            save_best_only=True,
            monitor="val_auc",
            mode="max",
            verbose=1,
        ),
        callbacks.CSVLogger(os.path.join(log_dir, "training_log.csv")),
    ]

    # ---------- STAGE 1: train top head ----------
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=cb_list,
        class_weight=class_weights,
    )

    # ---------- STAGE 2: fine-tune base ----------
    print("\n🔧 Fine-tuning base MobileNetV2 layers...")
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



# ---------- EVALUATION ----------

def evaluate_model(model, test_ds):
    print("\n🧪 Evaluating on test set...")

    # ALWAYS recompile with our desired metrics, ignore whatever was saved
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
    metric_names = model.metrics_names  # ['loss','accuracy','precision','recall','auc']

    print("\n📊 Test Metrics:")
    for name, value in zip(metric_names, results):
        if name in ["accuracy", "precision", "recall", "auc"]:
            print(f"  {name.capitalize():<10}: {value:.2%}")
        else:
            print(f"  {name.capitalize():<10}: {value:.4f}")

    # Save clean report
    lines = ["Model Evaluation Report", f"Date: {datetime.datetime.now()}"]
    for name, value in zip(metric_names, results):
        if name in ["accuracy", "precision", "recall", "auc"]:
            lines.append(f"{name}: {value:.4%}")
        else:
            lines.append(f"{name}: {value:.4f}")

    report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"📝 Evaluation report saved to {report_path}")
    return results



# ---------- MAIN ----------

if __name__ == "__main__":
    print("\n========================================")
    print("🚀 MULTI-CLASS CHEST X-RAY TRAINING")
    print("========================================\n")

    train_ds, val_ds, test_ds, class_names, class_counts, total = load_datasets()
    model = create_model(num_classes=len(class_names))
    h1, h2 = train_and_fine_tune(model, train_ds, val_ds, class_counts)
    evaluate_model(model, test_ds)
    model.save(MODEL_FILE)
    print(f"\n💾 Final model saved to {MODEL_FILE}")

    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
        for c in class_names:
            f.write(c + "\n")

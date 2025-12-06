import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# reuse dataset loader / paths from existing MobileNet script
from model import load_datasets, IMAGE_SIZE, MODEL_DIR

DENSENET_MODEL_FILE = os.path.join(MODEL_DIR, "multiclass_xray_densenet.keras")

EPOCHS = 15
FINE_TUNE_EPOCHS = 8

# ---------- FOCAL LOSS (better for imbalance) ----------
def focal_loss(gamma=2.0, alpha=0.25):
  def loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
  return loss

# ---------- MODEL ----------
def create_densenet_model(num_classes):
  print("\n🧠 Building DenseNet121-based model...")

  data_aug = tf.keras.Sequential(
    [
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.2),
      layers.RandomContrast(0.2),
    ],
    name="data_augmentation",
  )

  base_model = applications.DenseNet121(
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

  model = models.Model(inputs, outputs, name="multiclass_xray_densenet")

  model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=[
      tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
      tf.keras.metrics.Precision(name="precision"),
      tf.keras.metrics.Recall(name="recall"),
      tf.keras.metrics.AUC(name="auc", curve="PR"),
    ],
  )
  model.summary()
  return model

# ---------- TRAIN + FINE-TUNE ----------
def train_and_fine_tune(model, train_ds, val_ds):
  print("\n🎯 Training DenseNet with focal loss + class weights...")

  labels = train_ds.classes
  classes = np.unique(labels)
  class_weights_arr = compute_class_weight("balanced", classes=classes, y=labels)
  class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
  print(f"⚖️ Class weights: {class_weights}")

  log_dir = os.path.join(
    MODEL_DIR, "logs_densenet", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
      DENSENET_MODEL_FILE,
      save_best_only=True,
      monitor="val_auc",
      mode="max",
      verbose=1,
    ),
    callbacks.CSVLogger(os.path.join(log_dir, "training_log_densenet.csv")),
  ]

  history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=cb_list,
    class_weight=class_weights,
  )

  print("\n🔧 Fine-tuning DenseNet base layers...")
  base_model = model.get_layer(index=2)  # data_aug (0), densenet (1 or 2)
  base_model.trainable = True
  for layer in base_model.layers[:200]:
    layer.trainable = False

  model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=[
      tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
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

# ---------- EVAL ----------
def eval_model(model, test_ds):
  print("\n🧪 Evaluating DenseNet on test set...")
  results = model.evaluate(test_ds)
  names = model.metrics_names
  print("\n📊 DenseNet Test Metrics:")
  for n, v in zip(names, results):
    if n in ["accuracy", "precision", "recall", "auc"]:
      print(f"  {n:<10}: {v:.2%}")
    else:
      print(f"  {n:<10}: {v:.4f}")

if __name__ == "__main__":
  print("\n=======================================")
  print("🚀 DENSENET121 MULTI-CLASS TRAINING")
  print("=======================================\n")

  train_ds, val_ds, test_ds, class_names, class_counts, total = load_datasets()
  model = create_densenet_model(num_classes=len(class_names))
  h1, h2 = train_and_fine_tune(model, train_ds, val_ds)
  eval_model(model, test_ds)
  model.save(DENSENET_MODEL_FILE)
  print(f"\n💾 DenseNet model saved to {DENSENET_MODEL_FILE}")

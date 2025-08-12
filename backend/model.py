import os
import datetime
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks

# Suppress warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
# Use relative path to chest_xray directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chest_xray")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 15

# Ensure compatible paths
BASE_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_FILE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "pneumonia_model.keras")

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
def load_trained_model():
    """Load a pre-trained model from disk with enhanced checks"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found at: {MODEL_FILE}")
    
    try:
        # Custom objects for compatibility
        custom_objects = {
            'FixedDropout': layers.Dropout
        }
        
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Recompile for prediction
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model loaded from {MODEL_FILE}")
        print(f"Input shape: {model.input_shape}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# --------------------------------------------------
# DATA LOADING - Enhanced for paths
# --------------------------------------------------
def load_datasets():
    global BASE_DIR
    
    print("\n📂 Loading datasets...")
    
    # Verify paths
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing dataset split: {split_path}")
        print(f"✔ Found '{split}' at {split_path}")

    # Enhanced augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.mobilenet_v2.preprocess_input
    )

    train_ds = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_ds = test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'valid'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    test_ds = test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'test'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_ds.class_indices.keys())
    class_counts = np.bincount(train_ds.classes)
    total_samples = sum(class_counts)

    print(f"🏷️ Classes: {class_names}")
    print(f"📊 Class counts: {dict(zip(class_names, class_counts))}")
    print(f"🧮 Total samples: {total_samples}")

    return train_ds, val_ds, test_ds, class_names, class_counts

# --------------------------------------------------
# MODEL CREATION - Enhanced architecture
# --------------------------------------------------
def create_model(class_names, class_counts, total_samples):
    print("\n🧠 Building model...")

    # Enhanced augmentation
    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name='data_augmentation')

    # Base model with imagenet weights
    base_model = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=IMAGE_SIZE + (3,),
        pooling='avg'
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMAGE_SIZE + (3,), name='input_layer')
    x = data_aug(inputs)
    x = base_model(x, training=False)
    
    # Enhanced top layers
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation='softmax', name='output_layer')(x)

    model = tf.keras.Model(inputs, outputs, name='pneumonia_model')

    # Enhanced class weighting
    class_weights = {
        i: (total_samples / (len(class_names) * count)) ** 0.5
        for i, count in enumerate(class_counts)
    }
    print(f"⚖️ Class weights: {class_weights}")

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # REMOVED F1Score metric to fix the scalar shape error
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc', curve='PR')
        ]
    )
    model.summary()
    return model, class_weights

# --------------------------------------------------
# TRAINING - Enhanced with callbacks
# --------------------------------------------------
def train_model(model, train_ds, val_ds, class_weights):
    print("\n🎯 Training model...")

    log_dir = os.path.join(BASE_FILE_DIR, "logs", 
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # Enhanced callbacks
    cb_list = [
        callbacks.EarlyStopping(
            patience=8,
            monitor='val_auc',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_FILE,
            save_best_only=True,
            monitor='val_auc',  # Changed from 'val_f1_score'
            mode='max',
            save_weights_only=False,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),
        callbacks.CSVLogger(os.path.join(log_dir, 'training_log.csv'))
    ]

    # Train with steps per epoch
    steps_per_epoch = len(train_ds)
    validation_steps = len(val_ds)
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=cb_list,
        class_weight=class_weights
    )
    
    print("✅ Training complete")
    return history

# --------------------------------------------------
# FINE-TUNING - Gradual unfreezing
# --------------------------------------------------
def fine_tune_model(model, train_ds, val_ds, class_weights):
    print("\n🔧 Fine-tuning...")

    # Unfreeze layers gradually
    base_model = model.layers[2]
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    
    # Freeze all layers before `fine_tune_at`
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with lower learning rate
    # REMOVED F1Score metric to fix the scalar shape error
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            'precision',
            'recall',
            'auc'
        ]
    )

    # Train with smaller batches
    history_fine = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights
    )
    
    print("✅ Fine-tuning complete")
    return history_fine

# --------------------------------------------------
# EVALUATION - Comprehensive metrics
# --------------------------------------------------
def evaluate_model(model, test_ds):
    print("\n🧪 Evaluating on test set...")
    results = model.evaluate(test_ds)

    print("\n📊 Test Metrics:")
    print(f"  Loss:      {results[0]:.4f}")
    print(f"  Accuracy:  {results[1]:.2%}")
    print(f"  Precision: {results[2]:.2%}")
    print(f"  Recall:    {results[3]:.2%}")
    print(f"  AUC:       {results[4]:.2%}")
    
    # Save evaluation report
    report = (
        f"Model Evaluation Report\n"
        f"Date: {datetime.datetime.now()}\n"
        f"Loss: {results[0]:.4f}\n"
        f"Accuracy: {results[1]:.2%}\n"
        f"Precision: {results[2]:.2%}\n"
        f"Recall: {results[3]:.2%}\n"
        f"AUC: {results[4]:.2%}\n"
    )
    
    report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📝 Evaluation report saved to {report_path}")
    return results

# --------------------------------------------------
# PREDICTION - With uncertainty estimation
# --------------------------------------------------
def predict_image(model, img_path, class_names, num_samples=5):
    try:
        img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
        img_arr = tf.keras.utils.img_to_array(img)
        img_arr = applications.mobilenet_v2.preprocess_input(img_arr)
        
        # Create multiple augmented versions
        augmented_images = []
        for _ in range(num_samples):
            augmented = tf.image.random_flip_left_right(img_arr)
            augmented = tf.image.random_brightness(augmented, 0.1)
            augmented = tf.image.random_contrast(augmented, 0.9, 1.1)
            augmented_images.append(augmented)
            
        img_batch = tf.stack(augmented_images)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        
        # Combine predictions
        mean_preds = np.mean(predictions, axis=0)
        std_preds = np.std(predictions, axis=0)
        
        idx = np.argmax(mean_preds)
        predicted_class = class_names[idx]
        confidence = mean_preds[idx] * 100
        uncertainty = std_preds[idx] * 100
        
        prob_dict = {class_names[i]: {
            'probability': float(mean_preds[i]),
            'uncertainty': float(std_preds[i])
        } for i in range(len(class_names))}

        print(f"\n🖼️ {os.path.basename(img_path)} → {predicted_class} "
              f"({confidence:.2f}% ± {uncertainty:.2f}%)")
        
        return predicted_class, confidence, uncertainty, prob_dict

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None, None, None

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 PNEUMONIA DETECTION MODEL")
    print("="*50 + "\n")
    
    try:
        # Load data
        train_ds, val_ds, test_ds, class_names, class_counts = load_datasets()
        total_samples = sum(class_counts)
        
        # Build final model
        model, class_weights = create_model(class_names, class_counts, total_samples)
        
        # Train
        train_history = train_model(model, train_ds, val_ds, class_weights)
        fine_tune_history = fine_tune_model(model, train_ds, val_ds, class_weights)
        
        # Evaluate
        test_results = evaluate_model(model, test_ds)
        
        # Save final model
        model.save(MODEL_FILE)
        print(f"\n💾 Model saved → {MODEL_FILE}")

        # Sample predictions with uncertainty
        print("\n" + "="*50)
        print("🔍 SAMPLE PREDICTIONS WITH UNCERTAINTY ESTIMATION")
        print("="*50)
        
        for cls in class_names:
            class_dir = os.path.join(BASE_DIR, 'test', cls)
            if os.path.exists(class_dir) and os.listdir(class_dir):
                sample_path = os.path.join(class_dir, os.listdir(class_dir)[0])
                predict_image(model, sample_path, class_names)
                
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETE - READY FOR DEPLOYMENT")
        print("="*50)

    except FileNotFoundError as e:
        print(f"\n❌ Dataset Error: {e}")
        print("Expected structure:\n"
              f"  {BASE_DIR}/train/[NORMAL|PNEUMONIA]\n"
              f"  {BASE_DIR}/valid/[NORMAL|PNEUMONIA]\n"
              f"  {BASE_DIR}/test/[NORMAL|PNEUMONIA]\n")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
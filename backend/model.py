import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import numpy as np
import os
import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
BASE_DIR = os.path.abspath('../chest_xray')
IMAGE_SIZE = (224, 224)     # MobileNetV2 input size
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10

# --- Function to load the trained model ---
def load_trained_model():
    """Loads the pre-trained Keras model."""
    model_path = '../pneumonia_detection_model.keras'
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        # Print model summary for verification
        model.summary()
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("❗ Ensure the model file exists in the project root directory")
        raise

# --- Dataset Loading ---
def load_datasets():
    """Load and preprocess datasets with path verification"""
    print("\n📂 Loading datasets...")
    
    # Verify dataset paths
    required_dirs = ['train', 'valid', 'test']
    for dir_name in required_dirs:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Dataset directory not found: {dir_path}")
        print(f"✔ Found {dir_name} directory: {dir_path}")

    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.mobilenet_v2.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test data (only preprocessing)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=applications.mobilenet_v2.preprocess_input
    )
    
    # Create datasets
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
    
    # Get class names and counts
    class_names = list(train_ds.class_indices.keys())
    class_counts = np.bincount(train_ds.classes)
    total_samples = sum(class_counts)
    
    print(f"🏷️ Class names: {class_names}")
    print(f"📊 Class distribution: {dict(zip(class_names, class_counts))}")
    print(f"🧮 Total samples: {total_samples}")
    
    return train_ds, val_ds, test_ds, class_names, class_counts

# --- Model Creation ---
def create_model(class_names, class_counts, total_samples):
    """Create MobileNetV2-based model with transfer learning"""
    print("\n🧠 Creating model architecture...")
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name='data_augmentation')
    
    # Pre-trained base model
    base_model = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=IMAGE_SIZE + (3,),
        pooling='avg'
    )
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=IMAGE_SIZE + (3,), name='input_layer')
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.5, name='top_dropout')(x)
    outputs = layers.Dense(len(class_names), activation='softmax', name='output_layer')(x)
    
    model = tf.keras.Model(inputs, outputs, name='pneumonia_detection_model')
    
    # Calculate class weights
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = (1 / count) * (total_samples / len(class_names))
    
    print(f"⚖️ Class weights: {class_weights}")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    model.summary()
    return model, class_weights

# --- Training ---
def train_model(model, train_ds, val_ds, class_weights):
    """Train model with callbacks"""
    print("\n🎯 Starting model training...")
    
    # Create unique log directory
    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            patience=5,
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath='../pneumonia_detection_model.keras',
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            save_weights_only=False,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list,
        class_weight=class_weights
    )
    
    print("✅ Initial training complete")
    return history

# --- Fine-tuning ---
def fine_tune_model(model, train_ds, val_ds, class_weights):
    """Fine-tune top layers of base model"""
    print("\n🔧 Starting model fine-tuning...")
    
    # Get base model
    base_model = model.layers[2]  # Adjusted index for base model
    base_model.trainable = True
    
    # Set first 100 layers as non-trainable
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    # Train
    history_fine = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights
    )
    
    print("✅ Fine-tuning complete")
    return history_fine

# --- Evaluation ---
def evaluate_model(model, test_ds):
    """Evaluate model on test set"""
    print("\n🧪 Evaluating model on test set...")
    results = model.evaluate(test_ds)
    
    print("\n📊 Test Results:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.2%}")
    print(f"  Precision: {results[2]:.2%}")
    print(f"  Recall: {results[3]:.2%}")
    print(f"  AUC: {results[4]:.2%}")
    
    return results

# --- Prediction ---
def predict_image(model, img_path, class_names):
    """Make prediction on a single image"""
    try:
        # Load and preprocess image
        img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[predicted_class_index] * 100
        
        # Format probabilities
        probabilities = {class_names[i]: float(prob) for i, prob in enumerate(predictions)}
        
        print(f"\n🖼️ Image: {os.path.basename(img_path)}")
        print(f"🔮 Prediction: {predicted_class} ({confidence:.2f}% confidence)")
        for cls, prob in probabilities.items():
            print(f"  {cls}: {prob:.4f}")
        
        return predicted_class, confidence, probabilities
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None, None

# --- Main Execution ---
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 PNEUMONIA DETECTION MODEL TRAINING")
    print("="*50 + "\n")
    
    # Diagnostic information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Base dataset directory: {BASE_DIR}")
    
    # Load datasets
    try:
        train_ds, val_ds, test_ds, class_names, class_counts = load_datasets()
        total_samples = sum(class_counts)
        
        # Create model
        model, class_weights = create_model(class_names, class_counts, total_samples)
        
        # Train model
        history = train_model(model, train_ds, val_ds, class_weights)
        
        # Fine-tune model
        history_fine = fine_tune_model(model, train_ds, val_ds, class_weights)
        
        # Evaluate model
        test_results = evaluate_model(model, test_ds)
        
        # Save final model
        model.save('../pneumonia_detection_model.keras')
        print("\n💾 Model saved as '../pneumonia_detection_model.keras'")
        
        # Sample predictions
        print("\n" + "="*50)
        print("🔍 RUNNING SAMPLE PREDICTIONS")
        print("="*50)
        
        # Find sample images from each class
        sample_images = []
        for class_name in class_names:
            class_dir = os.path.join(BASE_DIR, 'test', class_name)
            if os.path.exists(class_dir) and os.listdir(class_dir):
                sample_image = os.path.join(class_dir, os.listdir(class_dir)[0])
                sample_images.append((class_name, sample_image))
        
        # Make predictions
        for class_name, img_path in sample_images:
            print(f"\nTesting {class_name} image...")
            predict_image(model, img_path, class_names)
        
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("❗ Please check your dataset directory structure:")
        print(f"  - Ensure the chest_xray folder exists at: {BASE_DIR}")
        print("  - It should contain 'train', 'valid', and 'test' subdirectories")
        print("  - Each subdirectory should have 'NORMAL' and 'PNEUMONIA' folders with images")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
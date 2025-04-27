# -*- coding: utf-8 -*-
"""
Plant Species and Disease Identification using Transfer Learning (Multi-Output CNN)
"""

import os
import glob
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from keras import layers, models, applications, optimizers, losses, metrics
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import tensorflow as tf

# --- Configuration ---
DATASET_PATH = 'dataset2/' # IMPORTANT: Change this to your dataset root folder
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50          # Initial training epochs
FINE_TUNE_EPOCHS = 20 # Epochs for fine-tuning
INIT_LR = 1e-3       # Initial Learning Rate
FINE_TUNE_LR = 1e-5  # Learning Rate for Fine-tuning

# --- 1. Data Loading and Preprocessing ---

def load_and_preprocess_data(dataset_path):
    """Loads images, extracts labels, preprocesses, encodes labels, and splits data."""
    image_paths = []
    species_labels = []
    disease_labels = []
    combined_labels = [] # For stratification

    print(f"[INFO] Searching for images in: {dataset_path}")
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path not found or is not a directory: {dataset_path}")

    # Iterate through subdirectories (expected format: species___disease)
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(subdir_path):
            # Extract labels from folder name (e.g., "Tomato___Late_Blight")
            try:
                # Ensure consistent splitting, handle potential extra underscores
                parts = subdir.split('___')
                if len(parts) >= 2:
                    species = parts[0]
                    disease = '___'.join(parts[1:]) # Join back if disease name had '___'
                else:
                    raise ValueError("Incorrect format")
            except ValueError:
                print(f"[WARNING] Skipping directory '{subdir}': Does not match 'Species___Disease' format.")
                continue

            # Find all images in the subdirectory
            img_files_found = glob.glob(os.path.join(subdir_path, '*.jpg')) + \
                              glob.glob(os.path.join(subdir_path, '*.JPG')) + \
                              glob.glob(os.path.join(subdir_path, '*.jpeg')) + \
                              glob.glob(os.path.join(subdir_path, '*.JPEG')) + \
                              glob.glob(os.path.join(subdir_path, '*.png')) + \
                              glob.glob(os.path.join(subdir_path, '*.PNG'))

            if not img_files_found:
                print(f"[WARNING] No image files found in directory: {subdir_path}")
                continue

            for image_file in img_files_found:
                image_paths.append(image_file)
                species_labels.append(species)
                disease_labels.append(disease)
                combined_labels.append(subdir) # Use folder name for stratification

    if not image_paths:
        raise ValueError(f"No valid images found in dataset path: {dataset_path}. Check subdirectories and file extensions.")

    print(f"[INFO] Found {len(image_paths)} images belonging to {len(set(combined_labels))} classes.")

    # --- Encode Labels ---
    species_binarizer = LabelBinarizer()
    disease_binarizer = LabelBinarizer()

    # Fit on the full set of labels found
    y_species_encoded = species_binarizer.fit_transform(species_labels)
    y_disease_encoded = disease_binarizer.fit_transform(disease_labels)

    num_species_classes = len(species_binarizer.classes_)
    num_disease_classes = len(disease_binarizer.classes_)
    print(f"[INFO] Number of species classes: {num_species_classes}")
    print(f"[INFO] Species classes: {species_binarizer.classes_}")
    print(f"[INFO] Number of disease classes: {num_disease_classes}")
    print(f"[INFO] Disease classes: {disease_binarizer.classes_}")

    # --- Load and Preprocess Images ---
    # NOTE: Loading all images into memory might be infeasible for very large datasets.
    # Consider using tf.data.Dataset with a generator function or
    # image_dataset_from_directory with custom label extraction for large datasets.
    print("[INFO] Loading and preprocessing images (this may take time)...")
    X_data = []
    y_species_list = []
    y_disease_list = []
    combined_labels_list = []
    valid_image_indices = []

    for i, img_path in enumerate(image_paths):
        try:
            img = cv2.imread(img_path)
            if img is None: # Check if image loading failed
                print(f"[WARNING] Failed to load image {img_path}. Skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB format
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            # Normalization will be done by ImageDataGenerator or Rescaling layer
            X_data.append(img)
            # Append corresponding labels only for successfully loaded images
            y_species_list.append(y_species_encoded[i])
            y_disease_list.append(y_disease_encoded[i])
            combined_labels_list.append(combined_labels[i])
            valid_image_indices.append(i) # Keep track of original index if needed elsewhere

        except Exception as e:
            print(f"[WARNING] Error loading or processing image {img_path}: {e}")

    # Convert lists of valid data to NumPy arrays
    X = np.array(X_data, dtype="float32")
    y_species_encoded = np.array(y_species_list)
    y_disease_encoded = np.array(y_disease_list)
    # combined_labels_list now holds the stratification labels for valid images

    print(f"[INFO] Successfully loaded {len(X)} images.")
    if len(X) == 0:
         raise ValueError("No images could be loaded successfully.")
    print(f"[INFO] Image data shape: {X.shape}")
    print(f"[INFO] Species label data shape: {y_species_encoded.shape}")
    print(f"[INFO] Disease label data shape: {y_disease_encoded.shape}")

    # --- Split Data ---
    # Use combined labels for stratification to keep class distribution similar
    print("[INFO] Splitting data into training, validation, and test sets...")
    # We use the combined_labels_list corresponding to the successfully loaded images (X)
    X_train_val, X_test, y_species_train_val, y_species_test, y_disease_train_val, y_disease_test, train_val_stratify, _ = train_test_split(
        X, y_species_encoded, y_disease_encoded, combined_labels_list,
        test_size=0.15, random_state=42, stratify=combined_labels_list
    )

    X_train, X_val, y_species_train, y_species_val, y_disease_train, y_disease_val = train_test_split(
        X_train_val, y_species_train_val, y_disease_train_val,
        test_size=0.1765, # 0.15 / (1 - 0.15) to get approx 15% of original for validation
        random_state=42, stratify=train_val_stratify # Stratify based on the labels in the train_val set
    )

    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    print(f"[INFO] Test samples: {len(X_test)}")

    # Package labels into dictionaries for multi-output model training
    y_train = {'species_output': y_species_train, 'disease_output': y_disease_train}
    y_val = {'species_output': y_species_val, 'disease_output': y_disease_val}
    y_test = {'species_output': y_species_test, 'disease_output': y_disease_test}


    return X_train, y_train, X_val, y_val, X_test, y_test, \
           species_binarizer, disease_binarizer, num_species_classes, num_disease_classes

# --- 2. Data Augmentation ---

# Using ImageDataGenerator for augmentation
# Normalize pixel values within the generator
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale validation and test data (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# --- 3. Model Building ---

def build_multi_output_model(num_species, num_diseases):
    """Builds the multi-output CNN model using transfer learning."""
    # Base model (MobileNetV2)
    base_model = applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Freeze base model layers initially

    # Input layer
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input_layer')

    # Pass input through base model
    # No need for separate Rescaling layer if using ImageDataGenerator with rescale=1./255
    # If not using ImageDataGenerator rescale, uncomment the Rescaling layer here.
    # x = layers.Rescaling(1./255)(inputs)
    x = base_model(inputs, training=False) # Use inputs directly

    # Custom classifier heads
    x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
    x = layers.Dropout(0.3, name='top_dropout_1')(x)
    # x = layers.Dense(256, activation='relu', name='intermediate_dense')(x) # Optional intermediate dense layer
    # x = layers.Dropout(0.3, name='top_dropout_2')(x)

    # Species output head
    species_out = layers.Dense(num_species, activation='softmax', name='species_output')(x)

    # Disease output head
    disease_out = layers.Dense(num_diseases, activation='softmax', name='disease_output')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=[species_out, disease_out], name='Plant_Classifier')

    return model

AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(ds, shuffle=False, augment=False):
    # Define augmentation layers (example)
    augmentation_layers = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest", interpolation="bilinear"),
    ], name="augmentation_pipeline")

    # Apply rescaling (do it first)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        # Use a buffer size related to your dataset size for good shuffling
        buffer_size = tf.data.experimental.cardinality(ds).numpy() # Get dataset size if possible
        if buffer_size == tf.data.UNKNOWN_CARDINALITY or buffer_size == tf.data.INFINITE_CARDINALITY:
             buffer_size = 1000 # Fallback buffer size
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Batch the dataset
    ds = ds.batch(BATCH_SIZE)

    # Apply augmentation after batching (or map before batching if needed)
    if augment:
        # Apply augmentation layers using map
        # Note: Augmentation layers expect batches by default usually
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching
    return ds.prefetch(buffer_size=AUTOTUNE)

# --- 4. Training ---

if __name__ == "__main__":
    # Load Data
    try:
        X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict, \
        species_binarizer, disease_binarizer, num_species, num_disease = load_and_preprocess_data(DATASET_PATH)

        # Save the binarizers for later use during prediction
        print("[INFO] Saving label binarizers...")
        with open('species_binarizer.pkl', 'wb') as f:
            pickle.dump(species_binarizer, f)
        with open('disease_binarizer.pkl', 'wb') as f:
            pickle.dump(disease_binarizer, f)

    except (ValueError, FileNotFoundError, IndexError) as e:
        print(f"[ERROR] Failed to load or process data: {e}")
        print("[INFO] Please ensure DATASET_PATH is set correctly, contains valid image files,")
        print("[INFO] and the directory structure is 'Species___Disease'.")
        exit(1) # Exit if data loading fails

    # --- Create data generators ---
    # Extract label arrays from the dictionaries
    y_species_train = y_train_dict['species_output']
    y_disease_train = y_train_dict['disease_output']
    y_species_val = y_val_dict['species_output']
    y_disease_val = y_val_dict['disease_output']
    y_species_test = y_test_dict['species_output']
    y_disease_test = y_test_dict['disease_output']

    # IMPORTANT: Create the target labels as a dictionary here,
    # matching the output layer names. tf.data handles this well.
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, {'species_output': y_species_train, 'disease_output': y_disease_train})
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, {'species_output': y_species_val, 'disease_output': y_disease_val})
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, {'species_output': y_species_test, 'disease_output': y_disease_test})
    )

    # Remove the old ImageDataGenerators
    # Use Keras augmentation layers within the dataset pipeline
    train_ds = configure_dataset(train_ds, shuffle=True, augment=True)
    val_ds = configure_dataset(val_ds) # No shuffle/augment for validation
    test_ds = configure_dataset(test_ds)  # No shuffle/augment for test

    # Build Model
    model = build_multi_output_model(num_species, num_disease)

    # Compile Model for Initial Training
    losses_dict = {
        'species_output': losses.CategoricalCrossentropy(),
        'disease_output': losses.CategoricalCrossentropy()
    }
    metrics_dict = {
        'species_output': metrics.CategoricalAccuracy(name='species_accuracy'),
        'disease_output': metrics.CategoricalAccuracy(name='disease_accuracy')
    }

    model.compile(
        optimizer=optimizers.Adam(learning_rate=INIT_LR),
        loss=losses_dict,      # Model compile still uses the dictionary
        metrics=metrics_dict   # Model compile still uses the dictionary
    )

    print("\n[INFO] Model Summary (Before Fine-tuning):")
    model.summary()

    # Callbacks (Keep as they are)
    checkpoint_path_init = "plant_model_init_best.keras"
    callbacks_init = [
        ModelCheckpoint(
            filepath=checkpoint_path_init,
            save_best_only=True,
            monitor='val_loss', # Monitor overall validation loss
            mode='min',         # Save on minimum validation loss
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,        # Number of epochs with no improvement after which training will be stopped.
            mode='min',
            restore_best_weights=True, # Restores model weights from the epoch with the best value of the monitored quantity.
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,         # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=5,         # Number of epochs with no improvement after which learning rate will be reduced.
            mode='min',
            verbose=1,
            min_lr=1e-6         # Lower bound on the learning rate.
        )
    ]

    # Initial Training (Train only the classifier heads)
    print("\n[INFO] Starting Initial Training...")
    # NO need for steps_per_epoch when using tf.data datasets
    history_init = model.fit(
        train_ds, # Pass the dataset directly
        epochs=EPOCHS,
        validation_data=val_ds, # Pass the dataset directly
        callbacks=callbacks_init,
        verbose=1
    )

    # --- 5. Fine-Tuning ---
    print("\n[INFO] Starting Fine-Tuning (Unfreezing Top Base Model Layers)...")

    # Load the best weights from initial training phase
    print(f"[INFO] Loading best weights from initial training: {checkpoint_path_init}")
    # Ensure the file exists before loading
    if os.path.exists(checkpoint_path_init):
         model.load_weights(checkpoint_path_init)
    else:
        print(f"[WARNING] Checkpoint file {checkpoint_path_init} not found. Proceeding with current weights.")


    # Find the base model layer (adjust index if model structure changes)
    base_model_layer = None
    for layer in model.layers:
        if layer.name == 'mobilenetv2_1.00_224': # Default name for MobileNetV2
             base_model_layer = layer
             break

    if base_model_layer is None:
        print("[ERROR] Could not find base model layer by name. Cannot proceed with fine-tuning.")
        exit(1)

    base_model_layer.trainable = True # Unfreeze the entire base model for fine-tuning

    # Decide how many layers to KEEP FROZEN (fine-tune layers AFTER this index)
    # This number depends heavily on the base model architecture (MobileNetV2 has ~154 layers)
    # Fine-tuning more layers requires lower learning rate and potentially more data
    fine_tune_from_layer = 100 # Example: Freeze first 100 layers, fine-tune the rest
    print(f"[INFO] Freezing the first {fine_tune_from_layer} layers of the base model.")

    for layer in base_model_layer.layers[:fine_tune_from_layer]:
         layer.trainable = False


    # Re-compile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR), # Very low LR
        loss=losses_dict,
        metrics=metrics_dict
    )

    print("\n[INFO] Model Summary (During Fine-tuning):")
    model.summary() # Verify trainable parameters changed

    # Define new checkpoint path for fine-tuning
    checkpoint_path_fine = "plant_model_fine_tuned_best.keras"
    # Create new callbacks or adjust existing ones for fine-tuning phase
    callbacks_fine = [
         ModelCheckpoint(
            filepath=checkpoint_path_fine,
            save_best_only=True,
            monitor='val_loss', # Monitor overall validation loss
            mode='min',
            verbose=1
        ),
        EarlyStopping( # You might want slightly different patience here
            monitor='val_loss',
            patience=15, # Allow more patience during fine-tuning
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau( # Reset patience or adjust
            monitor='val_loss',
            factor=0.2,
            patience=7,
            mode='min',
            verbose=1,
            min_lr=1e-7 # Can go even lower if needed
        )
    ]

    # Continue training (fine-tuning)
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    # Use the number of epochs already completed in history_init
    initial_epoch_fine_tune = len(history_init.epoch) if history_init and history_init.epoch else 0

    print(f"\n[INFO] Continuing training for fine-tuning up to {total_epochs} total epochs...")
    history_fine = model.fit(
        train_ds, # Pass the dataset directly
        epochs=total_epochs,
        initial_epoch=initial_epoch_fine_tune,
        validation_data=val_ds, # Pass the dataset directly
        callbacks=callbacks_fine,
        verbose=1
    )


    # --- 6. Evaluation ---
    print("\n[INFO] Evaluating final model on the test set...")

    # Load the absolute best weights saved during fine-tuning
    best_model_path = checkpoint_path_fine
    if not os.path.exists(best_model_path):
        print(f"[WARNING] Fine-tuning checkpoint {best_model_path} not found. Using initial training best weights {checkpoint_path_init} for evaluation.")
        best_model_path = checkpoint_path_init # Fallback to initial best if fine-tuning one isn't saved
        if not os.path.exists(best_model_path):
            print(f"[ERROR] No best model checkpoint found ({checkpoint_path_init} or {checkpoint_path_fine}). Cannot evaluate.")
            exit(1)

    print(f"[INFO] Loading best weights from: {best_model_path}")
    model.load_weights(best_model_path)

    print(f"[INFO] Loading best weights from: {best_model_path}")
    model.load_weights(best_model_path)

    # Evaluate using the test dataset
    test_results = model.evaluate(test_ds) # Pass the dataset directly

    # Print results (need to adjust index lookup slightly)
    # Keras might flatten the results list, check model.metrics_names order
    metrics_map = {name: i for i, name in enumerate(model.metrics_names)}
    print(f"\n[RESULTS] Test Loss (Overall): {test_results[metrics_map['loss']]:.4f}")
    print(f"[RESULTS] Test Loss (Species): {test_results[metrics_map['species_output_loss']]:.4f}")
    print(f"[RESULTS] Test Loss (Disease): {test_results[metrics_map['disease_output_loss']]:.4f}")
    print(f"[RESULTS] Test Accuracy (Species): {test_results[metrics_map['species_accuracy']]*100:.2f}%")
    print(f"[RESULTS] Test Accuracy (Disease): {test_results[metrics_map['disease_accuracy']]*100:.2f}%")


    print("\n[INFO] Generating Classification Reports...")
    # Predict on the test dataset
    # Note: model.predict on a tf.data.Dataset iterates through it.
    predictions = model.predict(test_ds)

    # True labels need to be extracted from the test_ds (or use original arrays)
    y_true_species_indices = np.argmax(y_species_test, axis=1)
    y_true_disease_indices = np.argmax(y_disease_test, axis=1)

    # Predictions are returned in a list matching model output order
    y_pred_species_indices = np.argmax(predictions[0], axis=1)
    y_pred_disease_indices = np.argmax(predictions[1], axis=1)

    # Ensure predicted length matches true length (in case of partial last batch)
    num_test_samples = len(X_test)
    y_pred_species_indices = y_pred_species_indices[:num_test_samples]
    y_pred_disease_indices = y_pred_disease_indices[:num_test_samples]


    print("\n--- Species Classification Report ---")
    try:
        print(classification_report(
            y_true_species_indices,
            y_pred_species_indices,
            target_names=species_binarizer.classes_,
            zero_division=0 # Handle cases with no predicted/true samples for a class
        ))
    except ValueError as e:
        print(f"[WARNING] Could not generate species classification report: {e}")


    print("\n--- Disease Classification Report ---")
    try:
        print(classification_report(
            y_true_disease_indices,
            y_pred_disease_indices,
            target_names=disease_binarizer.classes_,
            zero_division=0
        ))
    except ValueError as e:
        print(f"[WARNING] Could not generate disease classification report: {e}")
    

    # --- 7. Saving Final Model ---
    final_model_path = "plant_species_disease_model_final.keras"
    print(f"\n[INFO] Saving final trained model (loaded with best weights) to: {final_model_path}")
    model.save(final_model_path)

    print("\n[INFO] Script finished successfully.")
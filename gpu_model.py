import os
import glob
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, applications, optimizers, losses, metrics
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import time

# Configuration
DATASET_PATH = 'dataset2/'
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
# adjustable parameters
EPOCHS = 50          
FINE_TUNE_EPOCHS = 20
INIT_LR = 1e-3       
FINE_TUNE_LR = 1e-5  

# Data Loading and Preprocessing
def load_and_preprocess_data(dataset_path):
    """Loads images, extracts labels, preprocesses, encodes labels, and splits data."""
    start_time = time.time()
    print(f"[INFO] Starting data loading and preprocessing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    image_paths = []
    species_labels = []
    disease_labels = []
    combined_labels = []

    print(f"[INFO] Searching for images in: {os.path.abspath(dataset_path)}") # Show absolute path
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path not found or is not a directory: {os.path.abspath(dataset_path)}")

    # Iterate through subdirectories (expected format: species___disease)
    found_subdirs = 0
    skipped_subdirs = 0
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(subdir_path):
            found_subdirs += 1
            # Extract labels from folder name (e.g., "Tomato___Late_Blight")
            try:
                parts = subdir.split('___')
                if len(parts) >= 2:
                    species = parts[0].replace('_', ' ')
                    disease = '___'.join(parts[1:]).replace('_', ' ')
                else:
                    raise ValueError("Incorrect format")
            except ValueError:
                print(f"[WARNING] Skipping directory '{subdir}': Does not match 'Species___Disease' format.")
                skipped_subdirs +=1
                continue

            img_files_found = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
                 img_files_found.extend(glob.glob(os.path.join(subdir_path, ext)))
                 img_files_found.extend(glob.glob(os.path.join(subdir_path, ext.upper())))

            if not img_files_found:
                print(f"[WARNING] No image files found in directory: {subdir_path}")
                continue

            img_files_found = list(set(img_files_found))

            for image_file in img_files_found:
                image_paths.append(image_file)
                species_labels.append(species)
                disease_labels.append(disease)
                combined_labels.append(subdir)

    if skipped_subdirs > 0:
         print(f"[INFO] Processed {found_subdirs} directories, skipped {skipped_subdirs} due to naming format.")
    if not image_paths:
        raise ValueError(f"No valid images found in dataset path: {dataset_path}. Check subdirectories and file extensions.")

    print(f"[INFO] Found {len(image_paths)} images belonging to {len(set(combined_labels))} unique 'Species___Disease' combinations.")

    # Encode Labels
    species_binarizer = LabelBinarizer()
    disease_binarizer = LabelBinarizer()

    # Fit on the full set of labels found
    y_species_intermediate = species_binarizer.fit_transform(species_labels)
    y_disease_encoded = disease_binarizer.fit_transform(disease_labels)

    num_species_classes = len(species_binarizer.classes_)
    num_disease_classes = len(disease_binarizer.classes_)
    print(f"[INFO] Number of species classes found: {num_species_classes}")
    print(f"[INFO] Species classes: {species_binarizer.classes_}")
    print(f"[INFO] Number of disease classes found: {num_disease_classes}")
    print(f"[INFO] Disease classes: {disease_binarizer.classes_}")

    # Convert labels to one-hot if binary (for CategoricalCrossentropy)
    # (this logic correctly handles binary and multi-class cases)
    if num_species_classes == 2:
        print(f"[INFO] Converting binary species labels (shape {y_species_intermediate.shape}) to one-hot format...")
        if y_species_intermediate.ndim == 1:
             y_species_intermediate = y_species_intermediate.reshape(-1, 1)
        y_species_encoded = tf.keras.utils.to_categorical(y_species_intermediate, num_classes=2)
        print(f"[INFO] Species labels converted to shape: {y_species_encoded.shape}")
    elif num_species_classes == 1:
         raise ValueError("Found only one species class. Need at least two for classification.")
    else: # More than 2 classes
        y_species_encoded = y_species_intermediate

    if num_disease_classes == 2:
        print(f"[INFO] Converting binary disease labels (shape {y_disease_encoded.shape}) to one-hot format...")
        if y_disease_encoded.ndim == 1:
            y_disease_encoded = y_disease_encoded.reshape(-1, 1)
        y_disease_encoded = tf.keras.utils.to_categorical(y_disease_encoded, num_classes=2)
        print(f"[INFO] Disease labels converted to shape: {y_disease_encoded.shape}")
    elif num_disease_classes == 1:
        print(f"[WARNING] Found only one disease class: {disease_binarizer.classes_}. Ensure this is intended.")
        if y_disease_encoded.ndim == 1:
             y_disease_encoded = y_disease_encoded.reshape(-1, 1)
        # If LabelBinarizer outputs empty for single class (unlikely but possible)
        if y_disease_encoded.shape[1] == 0:
             print("[ERROR] Disease binarizer resulted in zero columns for single class. Adjusting.")
             y_disease_encoded = np.ones((len(y_disease_encoded), 1), dtype=int) # Make it (N, 1)
        # Model output layer will have 1 neuron, might need sigmoid if truly single class prediction
    else: # More than 2 classes
        y_disease_encoded = y_disease_encoded

    # Load and Preprocess Images
    print("[INFO] Loading and preprocessing images (this may take time)...")
    X_data = []
    y_species_list = []
    y_disease_list = []
    combined_labels_list = []
    processed_count = 0
    error_count = 0

    # Convert encoded labels to numpy for easier indexing in the loop
    y_species_encoded_np = y_species_encoded.numpy() if hasattr(y_species_encoded, 'numpy') else np.array(y_species_encoded)
    y_disease_encoded_np = y_disease_encoded.numpy() if hasattr(y_disease_encoded, 'numpy') else np.array(y_disease_encoded)

    total_images = len(image_paths)
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 500 == 0:
            print(f"[INFO] Processing image {i+1}/{total_images}...")
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Failed to load image (cv2.imread returned None): {img_path}. Skipping.")
                error_count += 1
                continue
            if img.size == 0:
                 print(f"[WARNING] Loaded empty image (size 0): {img_path}. Skipping.")
                 error_count += 1
                 continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            X_data.append(img)
            y_species_list.append(y_species_encoded_np[i])
            y_disease_list.append(y_disease_encoded_np[i])
            combined_labels_list.append(combined_labels[i])
            processed_count += 1

        except Exception as e:
            print(f"[WARNING] Error loading or processing image {img_path}: {e}")
            error_count += 1

    # Convert lists of valid data to NumPy arrays
    if not X_data:
         raise ValueError("No images could be loaded successfully. Check image files, paths, and permissions.")

    X = np.array(X_data, dtype="float32")
    y_species_final = np.array(y_species_list)
    y_disease_final = np.array(y_disease_list)

    print(f"[INFO] Successfully loaded and processed {processed_count} images.")
    if error_count > 0:
         print(f"[WARNING] Failed to load/process {error_count} images.")
    if len(combined_labels_list) != processed_count:
         raise RuntimeError(f"CRITICAL: Mismatch after processing - Images: {processed_count}, Combined Labels: {len(combined_labels_list)}")

    print(f"[INFO] Image data shape: {X.shape}")
    print(f"[INFO] Species label data shape: {y_species_final.shape}")
    print(f"[INFO] Disease label data shape: {y_disease_final.shape}")

    # Split Data
    print("[INFO] Splitting data into training, validation, and test sets...")
    # Stratify using the combined 'Species___Disease' label to keep distribution similar
    if len(combined_labels_list) == 0:
        print("[WARNING] No combined labels available for stratification. Splitting without stratification.")
        stratify_labels = None
    else:
        stratify_labels = combined_labels_list

    # First split: Create train+val and test sets (e.g., 85% train+val, 15% test)
    X_train_val, X_test, y_species_train_val, y_species_test, y_disease_train_val, y_disease_test, train_val_stratify, _ = train_test_split(
        X, y_species_final, y_disease_final, stratify_labels,
        test_size=0.15, random_state=42, stratify=stratify_labels # Use the combined label list for stratify
    )

    # Second split: Create train and validation sets from train+val
    # Target: Val size ~15% of original. Test is 15%. So Train+Val is 85%.
    # We need val_size = 0.15 / 0.85 = ~0.1765 of the train_val set.
    val_split_ratio = 0.15 / (1.0 - 0.15)

    if len(set(train_val_stratify)) < 2: # Check if stratification is possible for the second split
         print("[WARNING] Cannot stratify train/validation split (less than 2 classes in train_val_stratify). Splitting without stratification.")
         stratify_train_val = None
    else:
         stratify_train_val = train_val_stratify

    X_train, X_val, y_species_train, y_species_val, y_disease_train, y_disease_val = train_test_split(
        X_train_val, y_species_train_val, y_disease_train_val,
        test_size=val_split_ratio,
        random_state=42, stratify=stratify_train_val # Stratify this split too if possible
    )

    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    print(f"[INFO] Test samples: {len(X_test)}")

    y_test_dict = {'species_output': y_species_test, 'disease_output': y_disease_test}

    end_time = time.time()
    print(f"[INFO] Data loading and preprocessing finished in {end_time - start_time:.2f} seconds.")

    # Return individual arrays needed for the custom generator
    return X_train, y_species_train, y_disease_train, \
           X_val, y_species_val, y_disease_val, \
           X_test, y_test_dict, \
           species_binarizer, disease_binarizer, num_species_classes, num_disease_classes


# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Rescale pixel values to [0, 1]
    rotation_range=30,         # Random rotations
    width_shift_range=0.1,     # Random horizontal shifts
    height_shift_range=0.1,    # Random vertical shifts
    shear_range=0.2,           # Shear transformations
    zoom_range=0.2,            # Random zoom
    horizontal_flip=True,      # Random horizontal flips
    fill_mode='nearest'        # Strategy for filling newly created pixels
)
# IMPORTANT: Validation and Test data should ONLY be rescaled
val_test_datagen = ImageDataGenerator(rescale=1./255)

# sCustom Data Generator
def multi_output_data_generator(image_datagen, X, y_tuple, batch_size, shuffle=True):
    """
    Custom generator to handle multi-output labels with ImageDataGenerator augmentation.
    Yields batches of (augmented_images, (labels_output_1, labels_output_2, ...)).
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            # Check if batch_indices is empty (can happen if num_samples % batch_size != 0 and it's the last partial batch)
            if len(batch_indices) == 0:
                continue

            # Get the original data for the batch
            X_batch_orig = X[batch_indices]

            # Get the corresponding labels for the batch
            # Creates a tuple of label batches, e.g., (y_species_batch, y_disease_batch)
            try:
                 y_batch_tuple = tuple(y_arr[batch_indices] for y_arr in y_tuple)
            except IndexError as e:
                 print(f"[ERROR] IndexError in generator: {e}. Batch indices: {batch_indices}, y_tuple shapes: {[y.shape for y in y_tuple]}")
                 # Optionally skip this batch or re-raise
                 continue

            # Apply augmentation using the provided ImageDataGenerator
            # We use .flow() on the batch itself, asking it *not* to shuffle internally
            # and providing y=None as we handle labels manually.
            # It yields a single batch. Applies rescaling & augmentations defined in image_datagen.
            batch_augment_generator = image_datagen.flow(
                X_batch_orig,
                y=None, # Labels are handled manually
                batch_size=X_batch_orig.shape[0], # Process the whole batch at once
                shuffle=False # Keep order within the batch consistent with labels
            )
            X_batch_augmented = next(batch_augment_generator)

            # Sanity check: ensure augmented batch size matches label batch size
            if X_batch_augmented.shape[0] != y_batch_tuple[0].shape[0]:
                print(f"[ERROR] Generator batch size mismatch: X={X_batch_augmented.shape[0]}, y={y_batch_tuple[0].shape[0]} at indices {start}:{end}")
                continue

            yield X_batch_augmented, y_batch_tuple


# Model Building
def build_multi_output_model(num_species, num_diseases):
    """Builds the multi-output CNN model using transfer learning (MobileNetV2)."""
    print(f"[INFO] Building model for {num_species} species and {num_diseases} disease classes.")
    if num_diseases == 1:
         print("[WARNING] Building model with only 1 disease output class. Using softmax activation as configured.")
         # Consider sigmoid if binary classification is intended and labels are 0/1.
         # But since we forced one-hot (N, 1), softmax works technically.

    # Load MobileNetV2 pre-trained on ImageNet, without the top classification layer
    base_model = applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=False, # Exclude the final Dense layer of MobileNetV2
        weights='imagenet' # Use ImageNet pre-trained weights
    )
    # Freeze the layers of the base model initially
    base_model.trainable = False
    print(f"[INFO] Base model '{base_model.name}' loaded. Initial trainable status: {base_model.trainable}")

    # Define the input layer
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input_layer')

    # Pass inputs through the base model
    # training=False is important here since we froze the base model layers
    # It ensures layers like BatchNormalization run in inference mode
    x = base_model(inputs, training=False)

    # Add custom layers on top of the base model output
    # Global Average Pooling reduces spatial dimensions to 1x1xC
    x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
    # Dropout for regularization
    x = layers.Dropout(0.3, name='top_dropout')(x) # Slightly increased dropout

    species_output = layers.Dense(num_species, activation='softmax', name='species_output')(x)

    disease_output = layers.Dense(num_diseases, activation='softmax', name='disease_output')(x)

    # Create the final model
    model = models.Model(inputs=inputs, outputs=[species_output, disease_output], name='Plant_Classifier')

    return model

# Training
if __name__ == "__main__":
    print(f"[INFO] Starting script execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Using TensorFlow version: {tf.__version__}")

    # Load Data
    try:
        X_train, y_species_train, y_disease_train, \
        X_val, y_species_val, y_disease_val, \
        X_test, y_test_dict, \
        species_binarizer, disease_binarizer, num_species, num_disease = load_and_preprocess_data(DATASET_PATH)

        # Save the binarizers for later use (e.g., in prediction scripts)
        print("[INFO] Saving label binarizers...")
        os.makedirs("binarizers", exist_ok=True)
        with open('binarizers/species_binarizer.pkl', 'wb') as f:
            pickle.dump(species_binarizer, f)
        with open('binarizers/disease_binarizer.pkl', 'wb') as f:
            pickle.dump(disease_binarizer, f)
        print(f"[INFO] Binarizers saved to 'binarizers/' directory.")

    except (ValueError, FileNotFoundError, IndexError, RuntimeError) as e:
        print(f"[ERROR] Failed to load or process data: {e}")
        print("[INFO] Please ensure:")
        print(f"  - DATASET_PATH ('{DATASET_PATH}') points to the correct root folder.")
        print(f"  - The dataset folder contains subdirectories named 'Species___Disease'.")
        print(f"  - Subdirectories contain valid image files (jpg, png, etc.).")
        print(f"  - There are at least two different species classes.")
        print(f"  - Check previous warnings for skipped files or directories.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Create data generators using the custom generator function
    print(f"[INFO] Creating custom train_generator...")
    train_generator = multi_output_data_generator(
        train_datagen,      # Use the augmentation generator
        X_train,
        (y_species_train, y_disease_train), # Pass labels as a tuple
        BATCH_SIZE,
        shuffle=True
    )

    print(f"[INFO] Creating custom val_generator...")
    val_generator = multi_output_data_generator(
        val_test_datagen,   # Use the non-augmenting generator (only rescaling)
        X_val,
        (y_species_val, y_disease_val), # Pass labels as a tuple
        BATCH_SIZE,
        shuffle=False       # No shuffling for validation
    )

    # Rescale Test Data
    # It's crucial that test data is preprocessed the same way as validation data (rescaled)
    print("[INFO] Rescaling test data...")
    X_test_rescaled = X_test / 255.0 # Apply the same rescaling used in val_test_datagen

    # Build Model
    model = build_multi_output_model(num_species, num_disease)

    # Compile Model for Initial Training
    # Define loss functions for each output branch
    losses_dict = {
        'species_output': losses.CategoricalCrossentropy(),
        'disease_output': losses.CategoricalCrossentropy()
    }
    # Define metrics for each output branch
    metrics_dict = {
        'species_output': metrics.CategoricalAccuracy(name='species_accuracy'),
        'disease_output': metrics.CategoricalAccuracy(name='disease_accuracy')
    }
    # Define weights for each loss component if one task is more important (optional)
    # loss_weights_dict = {'species_output': 1.0, 'disease_output': 0.8}

    model.compile(
        optimizer=optimizers.Adam(learning_rate=INIT_LR), # Adam optimizer
        loss=losses_dict,            # Dictionary of losses
        # loss_weights=loss_weights_dict, # Optional loss weights
        metrics=metrics_dict         # Dictionary of metrics
    )
    print("\n[INFO] Model Summary (Before Fine-tuning):")
    model.summary(line_length=100)

    # Callbacks for Initial Training
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path_init = "checkpoints/plant_model_init_best.keras"
    callbacks_init = [
        # Save the best model based on validation loss
        ModelCheckpoint(filepath=checkpoint_path_init, save_best_only=True,
                        monitor='val_loss', mode='min', save_weights_only=False, verbose=1),
        # Stop training early if validation loss doesn't improve for 'patience' epochs
        EarlyStopping(monitor='val_loss', patience=10, mode='min', # Increased patience slightly
                      restore_best_weights=True, verbose=1), # Restore weights from the epoch with the best val_loss
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, # Reduce LR if no improvement for 5 epochs
                          mode='min', verbose=1, min_lr=1e-6) # Minimum learning rate
    ]

    # Initial Training
    print(f"\n[INFO] Starting Initial Training (Frozen Base Model) for {EPOCHS} epochs...")
    # Calculate steps per epoch, ensuring it's at least 1
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    validation_steps = max(1, len(X_val) // BATCH_SIZE)

    start_time_init = time.time()
    history_init = model.fit(
        train_generator,          # Custom training generator
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator, # Custom validation generator
        validation_steps=validation_steps,
        callbacks=callbacks_init,   # Callbacks for saving, early stopping, LR reduction
        verbose=1                   # Show progress bar
    )
    end_time_init = time.time()
    print(f"[INFO] Initial training finished in {end_time_init - start_time_init:.2f} seconds.")

    # Fine-Tuning
    print("\n[INFO] Starting Fine-Tuning Phase (Unfreezing Top Base Model Layers)...")

    # Load the best model saved during the initial training phase
    print(f"[INFO] Loading best model from initial training: {checkpoint_path_init}")
    if os.path.exists(checkpoint_path_init):
        # Load the entire model structure and weights
        # Need to re-compile after loading if changing trainability or optimizer
        model = tf.keras.models.load_model(checkpoint_path_init)
        print("[INFO] Successfully loaded best model from initial phase.")
    else:
        print(f"[WARNING] Checkpoint file '{checkpoint_path_init}' not found.")
        print("[INFO] Proceeding with model weights from the end of initial training (potentially suboptimal).")
        # If EarlyStopping restored weights, the 'model' variable holds them.

    # Find the base model layer to unfreeze
    base_model_layer = None
    for layer in model.layers:
        # Check name or type - MobileNetV2 usually has 'mobilenetv2' in its name
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
             base_model_layer = layer
             break

    history_fine = None # Initialize fine-tuning history

    if base_model_layer:
        print(f"[INFO] Found base model layer: '{base_model_layer.name}'.")

        # Unfreeze the base model
        base_model_layer.trainable = True
        print(f"[INFO] Base model '{base_model_layer.name}' unfrozen.")

        # Fine-tune only the top layers of the base model for stability
        # Example: Freeze the first 100 layers and fine-tune the rest
        fine_tune_at = 100
        print(f"[INFO] Freezing layers before layer {fine_tune_at} in '{base_model_layer.name}'.")

        if hasattr(base_model_layer, 'layers'): # Ensure it has sub-layers accessible
            for i, layer in enumerate(base_model_layer.layers[:fine_tune_at]):
                layer.trainable = False
            print(f"[INFO] First {fine_tune_at} layers frozen. Remaining layers are trainable.")
        else:
            print(f"[WARNING] Base model layer '{base_model_layer.name}' does not seem to have sub-layers accessible via '.layers'. Fine-tuning will affect the entire base model.")

        # Re-compile the model with a much lower learning rate for fine-tuning
        print(f"[INFO] Re-compiling model for fine-tuning with LR={FINE_TUNE_LR}.")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR), # Crucial: Lower LR
            loss=losses_dict,
            # loss_weights=loss_weights_dict, # Optional
            metrics=metrics_dict
        )
        print("\n[INFO] Model Summary (During Fine-tuning):")
        model.summary(line_length=100) # Note the change in trainable parameters

        # Callbacks for fine-tuning
        checkpoint_path_fine = "checkpoints/plant_model_fine_tuned_best.keras"
        callbacks_fine = [
            ModelCheckpoint(filepath=checkpoint_path_fine, save_best_only=True,
                            monitor='val_loss', mode='min', save_weights_only=False, verbose=1),
            # Longer patience might be needed for fine-tuning convergence
            EarlyStopping(monitor='val_loss', patience=15, mode='min',
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, # Slightly longer patience for LR reduction
                              mode='min', verbose=1, min_lr=1e-7) # Allow LR to go even lower
        ]

        # Determine starting epoch for fine-tuning history continuity
        # Use length of previous history if available and valid
        initial_epoch_fine_tune = 0
        if history_init and history_init.epoch:
             initial_epoch_fine_tune = history_init.epoch[-1] + 1
             print(f"[INFO] Resuming training from epoch {initial_epoch_fine_tune}.")
        else:
             # Fallback if initial training history is missing or empty
             initial_epoch_fine_tune = EPOCHS
             print(f"[WARNING] Initial training history not available. Starting fine-tuning epoch count from {initial_epoch_fine_tune}.")


        total_epochs_target = initial_epoch_fine_tune + FINE_TUNE_EPOCHS
        print(f"\n[INFO] Starting Fine-Tuning Training from epoch {initial_epoch_fine_tune} up to {total_epochs_target} total epochs...")

        start_time_fine = time.time()
        history_fine = model.fit(
            train_generator,          # Custom training generator
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs_target, # Total epochs across both phases
            initial_epoch=initial_epoch_fine_tune, # Start from where initial training left off
            validation_data=val_generator, # Custom validation generator
            validation_steps=validation_steps,
            callbacks=callbacks_fine,   # Callbacks specific to fine-tuning
            verbose=1
        )
        end_time_fine = time.time()
        print(f"[INFO] Fine-tuning finished in {end_time_fine - start_time_fine:.2f} seconds.")

    else:
        print("[ERROR] Could not find base model layer ('mobilenetv2'). Skipping fine-tuning phase.")
        checkpoint_path_fine = checkpoint_path_init # The best model is the one from init phase

    # Evaluation
    print("\n[INFO] Evaluating final model on the test set...")

    # Determine the best model checkpoint to load for final evaluation
    best_model_path = checkpoint_path_fine # Default to fine-tuned checkpoint
    if not os.path.exists(best_model_path) or history_fine is None: # If fine-tuning didn't run/save
         print(f"[INFO] Fine-tuning checkpoint '{best_model_path}' not found or fine-tuning skipped.")
         best_model_path = checkpoint_path_init # Fall back to initial training checkpoint
         if not os.path.exists(best_model_path):
             print(f"[ERROR] No best model checkpoint found ('{checkpoint_path_init}' or '{checkpoint_path_fine}').")
             print("[INFO] Evaluating model from the end of the last training phase (may be suboptimal).")
             # Keep the 'model' object as is (it holds weights from the end of the last training run)
         else:
             print(f"[INFO] Loading best model from initial training phase: '{best_model_path}'")
             model = tf.keras.models.load_model(best_model_path) # Load the best init model
    else:
        # If fine-tuning ran and saved a checkpoint, load that one
        print(f"[INFO] Loading best model from fine-tuning phase: '{best_model_path}'")
        model = tf.keras.models.load_model(best_model_path) # Load the best fine-tuned model

    # Perform Evaluation
    print("[INFO] Evaluating model performance on the rescaled test dataset...")
    # Ensure test data is rescaled (X_test_rescaled was prepared earlier)
    # Use the y_test_dict which has the correct dictionary format {'output_name': labels}
    test_results = model.evaluate(
        X_test_rescaled,
        y_test_dict,
        batch_size=BATCH_SIZE,
        verbose=0,            # Set to 1 for progress bar during evaluation
        return_dict=True      # Returns results as a dictionary
        )

    print("\n--- Test Set Evaluation Results ---")
    print(f"  Overall Test Loss: {test_results['loss']:.4f}")
    if 'species_accuracy' in test_results:
         print(f"  Species Test Accuracy: {test_results['species_accuracy']*100:.2f}%")
    if 'disease_accuracy' in test_results:
         print(f"  Disease Test Accuracy: {test_results['disease_accuracy']*100:.2f}%")
    # Print other metrics if needed (e.g., individual output losses)
    # for name, value in test_results.items():
    #      if name not in ['loss', 'species_accuracy', 'disease_accuracy']:
    #           print(f"  {name}: {value:.4f}")

    # Classification Reports
    print("\n[INFO] Generating Classification Reports for Test Set...")
    print("[INFO] Making predictions on the test set...")
    predictions = model.predict(X_test_rescaled, batch_size=BATCH_SIZE, verbose=1)

    # Extract true labels (already one-hot) from the y_test dictionary
    y_true_species = y_test_dict['species_output']
    y_true_disease = y_test_dict['disease_output']

    # Predictions is a list [pred_species, pred_disease] based on model output order
    y_pred_species_scores = predictions[0]
    y_pred_disease_scores = predictions[1]

    # Convert true labels and predictions from one-hot/scores to class indices (integers)
    y_true_species_indices = np.argmax(y_true_species, axis=1)
    y_pred_species_indices = np.argmax(y_pred_species_scores, axis=1)

    y_true_disease_indices = np.argmax(y_true_disease, axis=1)
    y_pred_disease_indices = np.argmax(y_pred_disease_scores, axis=1)

    # Get class names from the saved binarizers
    species_target_names = species_binarizer.classes_
    disease_target_names = disease_binarizer.classes_

    print(f"[DEBUG] Number of species target names: {len(species_target_names)}")
    print(f"[DEBUG] Number of disease target names: {len(disease_target_names)}")
    print(f"[DEBUG] Unique predicted species indices: {np.unique(y_pred_species_indices)}")
    print(f"[DEBUG] Unique predicted disease indices: {np.unique(y_pred_disease_indices)}")


    print("\n--- Species Classification Report ---")
    try:
        # Define the labels actually present or expected
        report_labels_species = np.arange(len(species_target_names))
        print(classification_report(
            y_true_species_indices,
            y_pred_species_indices,
            labels=report_labels_species, # Ensure report covers all known classes
            target_names=species_target_names,
            zero_division=0 # Avoid warnings for classes with no support in predictions
        ))
    except ValueError as e:
        print(f"[WARNING] Could not generate species classification report: {e}")
        print("Possible issues: Mismatch between labels found and binarizer classes, or issues with prediction shapes.")
        print(f"Shape y_true_species_indices: {y_true_species_indices.shape}")
        print(f"Shape y_pred_species_indices: {y_pred_species_indices.shape}")
        print(f"Unique true species labels: {np.unique(y_true_species_indices)}")
        print(f"Unique predicted species labels: {np.unique(y_pred_species_indices)}")


    print("\n--- Disease Classification Report ---")
    try:
        report_labels_disease = np.arange(len(disease_target_names))
        print(classification_report(
            y_true_disease_indices,
            y_pred_disease_indices,
            labels=report_labels_disease, # Ensure report covers all known classes
            target_names=disease_target_names,
            zero_division=0
        ))
    except ValueError as e:
        print(f"[WARNING] Could not generate disease classification report: {e}")
        print("Possible issues: Mismatch between labels found and binarizer classes, or issues with prediction shapes.")
        print(f"Shape y_true_disease_indices: {y_true_disease_indices.shape}")
        print(f"Shape y_pred_disease_indices: {y_pred_disease_indices.shape}")
        print(f"Unique true disease labels: {np.unique(y_true_disease_indices)}")
        print(f"Unique predicted disease labels: {np.unique(y_pred_disease_indices)}")

    # Saving Final Model
    # The 'model' variable should hold the best loaded weights at this point
    final_model_path = "plant_species_disease_model_final.keras"
    print(f"\n[INFO] Saving final trained model (loaded with best weights) to: {final_model_path}")
    try:
        model.save(final_model_path)
        print(f"[INFO] Final model successfully saved to {final_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save final model: {e}")
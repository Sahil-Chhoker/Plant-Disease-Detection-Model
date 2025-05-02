import os
import glob
import numpy as np
import cv2

from sklearn.model_selection import train_test_split # Potentially useful for custom splits, but not used here
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, applications, optimizers, losses, metrics
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import time
import matplotlib.pyplot as plt

# --- Configuration ---
DATASET_ROOT = 'archive/data/' # Root directory containing train, valid, test
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VALID_DIR = os.path.join(DATASET_ROOT, 'valid')
TEST_DIR = os.path.join(DATASET_ROOT, 'test')

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 30           # Epochs for initial training (frozen base)
FINE_TUNE_EPOCHS = 20 # Epochs for fine-tuning (unfrozen base)
INIT_LR = 1e-3        # Initial learning rate
FINE_TUNE_LR = 1e-5   # Fine-tuning learning rate
NUM_CLASSES = 2       # Binary classification: FHB vs Healthy

# --- Data Augmentation ---
# For Training Data: Apply various augmentations + rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Rescale pixel values to [0, 1]
    rotation_range=30,        # Random rotations
    width_shift_range=0.1,    # Random horizontal shifts
    height_shift_range=0.1,   # Random vertical shifts
    shear_range=0.2,          # Shear transformations
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True,     # Random horizontal flips
    fill_mode='nearest'       # Strategy for filling new pixels
)
# For Validation and Test Data: ONLY rescale pixel values
val_test_datagen = ImageDataGenerator(rescale=1./255)

# --- Model Building (Binary Classification with MobileNetV2) ---
def build_binary_classification_model():
    """Builds the binary classification CNN model using transfer learning (MobileNetV2)."""
    print(f"[INFO] Building model for binary classification.")

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
    # Set training=False when the base model is frozen
    x = base_model(inputs, training=False)

    # Add custom layers on top
    x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
    x = layers.Dropout(0.3, name='top_dropout')(x) # Regularization

    # Single output neuron with sigmoid activation for binary classification
    outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)

    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs, name='Wheat_FHB_Classifier')

    return model

# --- Function to plot training history ---
def plot_history(history_init, history_fine=None, save_path='training_history.png'):
    """Plots the training and validation accuracy and loss."""
    acc = history_init.history.get('accuracy', [])
    val_acc = history_init.history.get('val_accuracy', [])
    loss = history_init.history.get('loss', [])
    val_loss = history_init.history.get('val_loss', [])
    initial_epochs = len(acc)

    fine_epochs = 0
    if history_fine:
        acc += history_fine.history.get('accuracy', [])
        val_acc += history_fine.history.get('val_accuracy', [])
        loss += history_fine.history.get('loss', [])
        val_loss += history_fine.history.get('val_loss', [])
        fine_epochs = len(history_fine.history.get('accuracy', []))

    epochs_range = range(len(acc)) # Total epochs plotted

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    if history_fine and initial_epochs > 0 and fine_epochs > 0:
        # Plot a vertical line where fine-tuning starts
        plt.axvline(initial_epochs - 1, linestyle='--', color='r', label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    if history_fine and initial_epochs > 0 and fine_epochs > 0:
        # Plot a vertical line where fine-tuning starts
        plt.axvline(initial_epochs - 1, linestyle='--', color='r', label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Training history plot saved as '{save_path}'")
    # plt.show() # Uncomment to display the plot interactively

# --- Main Training Script ---
if __name__ == "__main__":
    print(f"[INFO] Starting script execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Using TensorFlow version: {tf.__version__}")

    # --- GPU Check ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"[INFO] {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected and configured.")
            print("[INFO] IMPORTANT: Ensure your TensorFlow version is compatible with your CUDA installation (CUDA 12.3 likely needs TF 2.15+).")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"[ERROR] Could not configure GPU memory growth: {e}")
            print("[INFO] Proceeding with default GPU configuration.")
    else:
        print("[WARNING] No GPU detected by TensorFlow. Training will occur on the CPU (potentially much slower).")
        print("[INFO] If you have a GPU, check TensorFlow installation (use 'tensorflow', not 'tensorflow-cpu'), NVIDIA drivers, and CUDA/cuDNN setup.")


    print(f"[INFO] Dataset Root: {os.path.abspath(DATASET_ROOT)}")

    # Verify directories exist
    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] Training directory not found: {TRAIN_DIR}")
        exit(1)
    if not os.path.isdir(VALID_DIR):
        print(f"[ERROR] Validation directory not found: {VALID_DIR}")
        exit(1)
    if not os.path.isdir(TEST_DIR):
        print(f"[ERROR] Test directory not found: {TEST_DIR}")
        exit(1)

    # --- Create Data Generators ---
    print("[INFO] Creating data generators...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary', # Crucial for BinaryCrossentropy loss
        color_mode='rgb',    # Ensure 3 channels for MobileNetV2
        shuffle=True         # Shuffle training data
    )

    validation_generator = val_test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False        # No need to shuffle validation data
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE, # Can adjust for evaluation if memory allows
        class_mode='binary',
        color_mode='rgb',
        shuffle=False        # DO NOT shuffle test data for evaluation
    )

    # Check class indices (important for interpreting results)
    print(f"[INFO] Class Indices found by generators:")
    print(f"  Train: {train_generator.class_indices}")
    print(f"  Validation: {validation_generator.class_indices}")
    print(f"  Test: {test_generator.class_indices}")
    # Ensure they are consistent, e.g., {'Fusarium Head Blight': 0, 'Healthy': 1} or vice-versa
    if not (train_generator.class_indices == validation_generator.class_indices == test_generator.class_indices):
         print("[ERROR] Class indices mismatch between data splits. Check directory structure.")
         exit(1)
    class_names = list(train_generator.class_indices.keys())
    print(f"[INFO] Class names: {class_names}")
    num_train_samples = train_generator.samples
    num_val_samples = validation_generator.samples
    num_test_samples = test_generator.samples
    print(f"[INFO] Found {num_train_samples} training, {num_val_samples} validation, {num_test_samples} test images.")

    # --- Build Model ---
    model = build_binary_classification_model()

    # --- Compile Model for Initial Training ---
    print("[INFO] Compiling model for initial training phase...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=INIT_LR),
        loss=losses.BinaryCrossentropy(), # Loss for binary (0/1) classification
        metrics=[metrics.BinaryAccuracy(name='accuracy')] # Use BinaryAccuracy
    )
    print("\n[INFO] Model Summary (Before Fine-tuning):")
    model.summary(line_length=100)

    # --- Callbacks for Initial Training ---
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path_init = "checkpoints/wheat_model_init_best.keras" # Use .keras format
    callbacks_init = [
        ModelCheckpoint(filepath=checkpoint_path_init, save_best_only=True,
                        monitor='val_accuracy', mode='max', # Save based on max validation accuracy
                        save_weights_only=False, verbose=1), # Save entire model
        EarlyStopping(monitor='val_loss', patience=10, mode='min',
                      restore_best_weights=True, verbose=1), # Stop if val_loss doesn't improve
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                          mode='min', verbose=1, min_lr=1e-6) # Reduce LR if val_loss plateaus
    ]

    # --- Initial Training (Base Model Frozen) ---
    print(f"\n[INFO] Starting Initial Training (Frozen Base Model) for up to {EPOCHS} epochs...")
    start_time_init = time.time()
    history_init = model.fit(
        train_generator,
        # steps_per_epoch=num_train_samples // BATCH_SIZE, # Usually inferred
        epochs=EPOCHS,
        validation_data=validation_generator,
        # validation_steps=num_val_samples // BATCH_SIZE, # Usually inferred
        callbacks=callbacks_init,
        verbose=1
    )
    end_time_init = time.time()
    print(f"[INFO] Initial training finished in {end_time_init - start_time_init:.2f} seconds.")

    # --- Fine-Tuning Phase ---
    print("\n[INFO] Preparing for Fine-Tuning Phase...")

    # Load the best model saved during the initial training phase
    # This ensures we start fine-tuning from the best initial weights
    print(f"[INFO] Loading best model from initial training: {checkpoint_path_init}")
    if os.path.exists(checkpoint_path_init):
        # Load the entire model (including optimizer state if saved)
        model = tf.keras.models.load_model(checkpoint_path_init)
        print("[INFO] Successfully loaded best model from initial phase.")
    else:
        print(f"[WARNING] Checkpoint file '{checkpoint_path_init}' not found.")
        print("[INFO] Proceeding with model weights from the end of initial training.")
        # If EarlyStopping restored best weights, the current 'model' object has them.
        # If not, we proceed with the weights from the last epoch of initial training.

    # Find the base model layer within the loaded model
    base_model_layer = None
    for layer in model.layers:
        # Check name or type if name is uncertain
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
             base_model_layer = layer
             break

    history_fine = None # Initialize fine-tuning history

    if base_model_layer:
        print(f"[INFO] Found base model layer: '{base_model_layer.name}'.")
        base_model_layer.trainable = True # Unfreeze the base model
        print(f"[INFO] Base model '{base_model_layer.name}' unfrozen for fine-tuning.")

        # Fine-tune only the top layers of the base model
        # MobileNetV2 has 154 layers. Let's fine-tune from layer 100 onwards.
        fine_tune_at = 100
        print(f"[INFO] Freezing layers before layer index {fine_tune_at} in '{base_model_layer.name}'.")

        if hasattr(base_model_layer, 'layers'):
            num_base_layers = len(base_model_layer.layers)
            if fine_tune_at >= num_base_layers:
                 print(f"[WARNING] fine_tune_at ({fine_tune_at}) >= number of layers ({num_base_layers}). Fine-tuning all base layers.")
                 fine_tune_at = 0 # Adjust to freeze none if index is out of bounds

            for i, layer in enumerate(base_model_layer.layers[:fine_tune_at]):
                layer.trainable = False
                # print(f"Layer {i}: {layer.name} frozen") # Uncomment for debugging

            print(f"[INFO] First {fine_tune_at} layers of base model frozen. Remaining {num_base_layers - fine_tune_at} are trainable.")
        else:
            print(f"[WARNING] Could not access internal layers of '{base_model_layer.name}'. Fine-tuning the entire base model block.")

        # Re-compile the model with a lower learning rate for fine-tuning
        print(f"[INFO] Re-compiling model for fine-tuning with LR={FINE_TUNE_LR}.")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR), # Very low LR for fine-tuning
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.BinaryAccuracy(name='accuracy')]
        )
        print("\n[INFO] Model Summary (During Fine-tuning):")
        model.summary(line_length=100)

        # Callbacks for fine-tuning
        checkpoint_path_fine = "checkpoints/wheat_model_fine_tuned_best.keras"
        callbacks_fine = [
            ModelCheckpoint(filepath=checkpoint_path_fine, save_best_only=True,
                            monitor='val_accuracy', mode='max', # Save best based on val accuracy
                            save_weights_only=False, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, mode='min', # More patience for fine-tuning
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, # More patience for LR reduction
                              mode='min', verbose=1, min_lr=1e-7) # Allow even lower LR
        ]

        # Determine the starting epoch for the fine-tuning phase for history continuity
        # Use the number of epochs actually run in the initial phase
        actual_initial_epochs = len(history_init.epoch) if history_init and history_init.epoch else EPOCHS
        print(f"[INFO] Initial training ran for {actual_initial_epochs} epochs.")

        total_epochs_target = actual_initial_epochs + FINE_TUNE_EPOCHS
        print(f"\n[INFO] Starting Fine-Tuning Training from effective epoch {actual_initial_epochs} up to {total_epochs_target} total epochs...")

        start_time_fine = time.time()
        history_fine = model.fit(
            train_generator,
            epochs=total_epochs_target,
            initial_epoch=actual_initial_epochs, # Start epoch numbering after initial training
            validation_data=validation_generator,
            callbacks=callbacks_fine,
            verbose=1
        )
        end_time_fine = time.time()
        print(f"[INFO] Fine-tuning finished in {end_time_fine - start_time_fine:.2f} seconds.")

    else:
        print("[ERROR] Could not find base model layer ('mobilenetv2'). Skipping fine-tuning phase.")
        # If fine-tuning is skipped, the best model is the one from the initial phase
        checkpoint_path_fine = checkpoint_path_init

    # --- Plot Training History ---
    print("[INFO] Plotting training history...")
    plot_history(history_init, history_fine) # Pass both histories

    # --- Evaluation on Test Set ---
    print("\n[INFO] Evaluating final model on the test set...")

    # Determine the path of the best model (prefer fine-tuned if available and successful)
    best_model_path = checkpoint_path_init # Default to initial best
    if history_fine and os.path.exists(checkpoint_path_fine):
        print(f"[INFO] Using best model from fine-tuning phase: '{checkpoint_path_fine}'")
        best_model_path = checkpoint_path_fine
    elif os.path.exists(checkpoint_path_init):
         print(f"[INFO] Fine-tuning skipped or checkpoint not found. Using best model from initial phase: '{checkpoint_path_init}'")
    else:
        print(f"[ERROR] No best model checkpoint found at '{checkpoint_path_init}' or '{checkpoint_path_fine}'.")
        print("[INFO] Evaluating model from the end of the last training phase (may be suboptimal).")
        best_model_path = None # Use the model currently in memory

    # Load the selected best model for final evaluation
    if best_model_path and os.path.exists(best_model_path):
        print(f"[INFO] Loading model for evaluation from: {best_model_path}")
        # Ensure custom objects are handled if necessary, though MobileNetV2 usually doesn't require it
        # For simple models like this, tf.keras.models.load_model is often sufficient
        model = tf.keras.models.load_model(best_model_path)
        print("[INFO] Model loaded successfully for evaluation.")
    elif not best_model_path:
         print("[INFO] Evaluating the model currently in memory.")
    else:
        print(f"[ERROR] Failed to load the model from {best_model_path}. Aborting evaluation.")
        exit(1)


    # Perform Evaluation using the test generator
    print("[INFO] Evaluating model performance on the test dataset...")
    test_loss, test_accuracy = model.evaluate(
        test_generator,
        # steps=num_test_samples // BATCH_SIZE + (1 if num_test_samples % BATCH_SIZE else 0), # Usually inferred
        verbose=1
    )

    print("\n--- Test Set Evaluation Results ---")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")


    # --- Classification Report ---
    print("\n[INFO] Generating Classification Report for Test Set...")
    # Predict probabilities on the test set
    # Ensure the generator is reset and predict over all test samples exactly once
    test_generator.reset() # Reset generator before predicting
    steps = int(np.ceil(num_test_samples / BATCH_SIZE))
    print(f"[INFO] Making predictions on {num_test_samples} test samples using {steps} steps...")

    predictions_prob = model.predict(test_generator, steps=steps, verbose=1)

    # Get true labels (0 or 1) directly from the generator
    y_true = test_generator.classes
    # Ensure we only use labels corresponding to the actual samples, not padding from batches
    y_true = y_true[:num_test_samples]


    # Convert probabilities to class labels (0 or 1) using a 0.5 threshold
    # predictions_prob will be shape (num_samples, 1) for sigmoid output
    y_pred = (predictions_prob > 0.5).astype(int).flatten() # Flatten to 1D array

    # Ensure we have the correct number of predictions
    if len(y_pred) != num_test_samples:
        print(f"[WARNING] Number of predictions ({len(y_pred)}) does not match number of test samples ({num_test_samples}). Adjusting prediction array.")
        # This usually happens if steps * BATCH_SIZE > num_samples
        y_pred = y_pred[:num_test_samples]

    print("\n--- Classification Report ---")
    # Use the class names derived from the generator's class_indices
    try:
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names, # Use the names learned by the generator
            zero_division=0 # Avoid warnings for classes with no predicted samples
        )
        print(report)
    except ValueError as e:
        print(f"[ERROR] Could not generate classification report: {e}")
        print("Ensure y_true and y_pred contain valid binary labels and target_names match.")
        print(f"y_true unique values: {np.unique(y_true)}")
        print(f"y_pred unique values: {np.unique(y_pred)}")
        print(f"target_names: {class_names}")


    # --- Save the Final Best Model ---
    final_model_save_path = "final_wheat_model.keras" # Save in modern Keras format
    if best_model_path and os.path.exists(best_model_path):
        print(f"[INFO] Saving the final best evaluated model from {best_model_path} to: {final_model_save_path}")
        # Re-load the best model before saving, just to be certain
        model_to_save = tf.keras.models.load_model(best_model_path)
        model_to_save.save(final_model_save_path)
        print("[INFO] Final model saved successfully.")
    elif not best_model_path:
         print(f"[INFO] Saving the model currently in memory (from end of last training phase) to: {final_model_save_path}")
         model.save(final_model_save_path)
         print("[INFO] Final model saved successfully.")
    else:
         print("[WARNING] Could not save final model as the best checkpoint path was invalid.")


    print(f"\n[INFO] Script finished execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
import os
import glob
import numpy as np
import cv2
# import pickle # No longer needed
from sklearn.model_selection import train_test_split # Potentially useful for custom splits, but not used here
# LabelBinarizer not needed with flow_from_directory and class_mode='categorical'
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, applications, optimizers, losses, metrics
# Correct import path for newer TF/Keras versions (as specified by user)
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
# <<< CHANGE: Updated for 3 classes based on the image >>>
NUM_CLASSES = 3       # Multi-class classification: FHB vs Healthy vs Leaf Blight

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

# <<< CHANGE: Renamed function and updated docstring >>>
# --- Model Building (Multi-Class Classification with MobileNetV2) ---
def build_multi_class_model(num_classes):
    """Builds the multi-class classification CNN model using transfer learning (MobileNetV2)."""
    # <<< CHANGE: Updated print statement >>>
    print(f"[INFO] Building model for {num_classes}-class classification.")

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

    # <<< CHANGE: Output layer for multi-class classification >>>
    # Use num_classes neurons with softmax activation
    outputs = layers.Dense(num_classes, activation='softmax', name='output_layer')(x)

    # Create the final model
    # <<< CHANGE: Updated model name slightly >>>
    model = models.Model(inputs=inputs, outputs=outputs, name='Wheat_Disease_Classifier')

    return model

# --- Function to plot training history ---
# (No changes needed in plot_history function itself, but metric names might change slightly in history dict)
def plot_history(history_init, history_fine=None, save_path='training_history_multi_class.png'):
    """Plots the training and validation accuracy and loss."""
    # Use .get() to handle potential missing keys gracefully
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
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"[INFO] {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected and configured.")
            print("[INFO] IMPORTANT: Ensure your TensorFlow version is compatible with your CUDA installation (CUDA 12.3 likely needs TF 2.15+).")
        except RuntimeError as e:
            print(f"[ERROR] Could not configure GPU memory growth: {e}")
            print("[INFO] Proceeding with default GPU configuration.")
    else:
        print("[WARNING] No GPU detected by TensorFlow. Training will occur on the CPU (potentially much slower).")
        print("[INFO] If you have a GPU, check TensorFlow installation, NVIDIA drivers, and CUDA/cuDNN setup.")

    print(f"[INFO] Dataset Root: {os.path.abspath(DATASET_ROOT)}")
    print("[INFO] IMPORTANT: Ensure class subdirectories are named identically in train/, valid/, and test/.")
    print("[INFO] Expected class names (based on train/): Fusarium Head Blight, Healthy, Leaf Blight")


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
        # <<< CHANGE: Updated for multi-class >>>
        class_mode='categorical', # Use 'categorical' for multi-class one-hot labels
        color_mode='rgb',
        shuffle=True
    )

    validation_generator = val_test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        # <<< CHANGE: Updated for multi-class >>>
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        # <<< CHANGE: Updated for multi-class >>>
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    # Check class indices
    print(f"[INFO] Class Indices found by generators:")
    print(f"  Train: {train_generator.class_indices}")
    print(f"  Validation: {validation_generator.class_indices}")
    print(f"  Test: {test_generator.class_indices}")

    # Check for consistency and correct number of classes
    if not (train_generator.class_indices == validation_generator.class_indices == test_generator.class_indices):
         print("[ERROR] Class indices mismatch between data splits. Check directory structure and naming consistency!")
         exit(1)
    if len(train_generator.class_indices) != NUM_CLASSES:
         print(f"[ERROR] Generator found {len(train_generator.class_indices)} classes, but NUM_CLASSES is set to {NUM_CLASSES}. Check directories or NUM_CLASSES setting.")
         exit(1)

    class_names = list(train_generator.class_indices.keys())
    print(f"[INFO] Class names: {class_names}")
    num_train_samples = train_generator.samples
    num_val_samples = validation_generator.samples
    num_test_samples = test_generator.samples
    print(f"[INFO] Found {num_train_samples} training, {num_val_samples} validation, {num_test_samples} test images.")

    # --- Build Model ---
    # <<< CHANGE: Pass NUM_CLASSES to the updated build function >>>
    model = build_multi_class_model(num_classes=NUM_CLASSES)

    # --- Compile Model for Initial Training ---
    print("[INFO] Compiling model for initial training phase...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=INIT_LR),
        # <<< CHANGE: Updated loss for multi-class >>>
        loss=losses.CategoricalCrossentropy(), # Loss for multi-class one-hot labels
        # <<< CHANGE: Updated metric for multi-class >>>
        metrics=[metrics.CategoricalAccuracy(name='accuracy')] # Use CategoricalAccuracy
    )
    print("\n[INFO] Model Summary (Before Fine-tuning):")
    model.summary(line_length=100)

    # --- Callbacks for Initial Training ---
    os.makedirs("checkpoints", exist_ok=True)
    # <<< CHANGE: Updated checkpoint file name >>>
    checkpoint_path_init = "checkpoints/wheat_multi_class_init_best.keras"
    callbacks_init = [
        ModelCheckpoint(filepath=checkpoint_path_init, save_best_only=True,
                        monitor='val_accuracy', mode='max', # Monitor validation accuracy (renamed metric)
                        save_weights_only=False, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, mode='min',
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                          mode='min', verbose=1, min_lr=1e-6)
    ]

    # --- Initial Training (Base Model Frozen) ---
    print(f"\n[INFO] Starting Initial Training (Frozen Base Model) for up to {EPOCHS} epochs...")
    start_time_init = time.time()
    history_init = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks_init,
        verbose=1
    )
    end_time_init = time.time()
    print(f"[INFO] Initial training finished in {end_time_init - start_time_init:.2f} seconds.")

    # --- Fine-Tuning Phase ---
    # <<< CHANGE: Updated print statement >>>
    print("\n[INFO] Preparing for Fine-Tuning Phase (Multi-Class)...")

    # Load the best model saved during the initial training phase
    print(f"[INFO] Loading best model from initial training: {checkpoint_path_init}")
    if os.path.exists(checkpoint_path_init):
        model = tf.keras.models.load_model(checkpoint_path_init)
        print("[INFO] Successfully loaded best model from initial phase.")
    else:
        print(f"[WARNING] Checkpoint file '{checkpoint_path_init}' not found.")
        print("[INFO] Proceeding with model weights from the end of initial training.")

    # Find the base model layer
    base_model_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
             base_model_layer = layer
             break

    history_fine = None

    if base_model_layer:
        print(f"[INFO] Found base model layer: '{base_model_layer.name}'.")
        base_model_layer.trainable = True
        print(f"[INFO] Base model '{base_model_layer.name}' unfrozen for fine-tuning.")

        # Fine-tune only the top layers
        fine_tune_at = 100
        print(f"[INFO] Freezing layers before layer index {fine_tune_at} in '{base_model_layer.name}'.")

        if hasattr(base_model_layer, 'layers'):
            num_base_layers = len(base_model_layer.layers)
            if fine_tune_at >= num_base_layers:
                 print(f"[WARNING] fine_tune_at ({fine_tune_at}) >= number of layers ({num_base_layers}). Fine-tuning all base layers.")
                 fine_tune_at = 0
            for i, layer in enumerate(base_model_layer.layers[:fine_tune_at]):
                layer.trainable = False
            print(f"[INFO] First {fine_tune_at} layers of base model frozen. Remaining {num_base_layers - fine_tune_at} are trainable.")
        else:
            print(f"[WARNING] Could not access internal layers of '{base_model_layer.name}'. Fine-tuning the entire base model block.")

        # Re-compile the model with lower LR for fine-tuning
        print(f"[INFO] Re-compiling model for fine-tuning with LR={FINE_TUNE_LR}.")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
            # <<< CHANGE: Ensure correct loss and metrics are used for re-compile >>>
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.CategoricalAccuracy(name='accuracy')]
        )
        print("\n[INFO] Model Summary (During Fine-tuning):")
        model.summary(line_length=100)

        # Callbacks for fine-tuning
        # <<< CHANGE: Updated checkpoint file name >>>
        checkpoint_path_fine = "checkpoints/wheat_multi_class_fine_tuned_best.keras"
        callbacks_fine = [
            ModelCheckpoint(filepath=checkpoint_path_fine, save_best_only=True,
                            monitor='val_accuracy', mode='max',
                            save_weights_only=False, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, mode='min',
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7,
                              mode='min', verbose=1, min_lr=1e-7)
        ]

        # Determine starting epoch for continuity
        actual_initial_epochs = len(history_init.epoch) if history_init and history_init.epoch else EPOCHS
        print(f"[INFO] Initial training ran for {actual_initial_epochs} epochs.")

        total_epochs_target = actual_initial_epochs + FINE_TUNE_EPOCHS
        print(f"\n[INFO] Starting Fine-Tuning Training from effective epoch {actual_initial_epochs} up to {total_epochs_target} total epochs...")

        start_time_fine = time.time()
        history_fine = model.fit(
            train_generator,
            epochs=total_epochs_target,
            initial_epoch=actual_initial_epochs,
            validation_data=validation_generator,
            callbacks=callbacks_fine,
            verbose=1
        )
        end_time_fine = time.time()
        print(f"[INFO] Fine-tuning finished in {end_time_fine - start_time_fine:.2f} seconds.")

    else:
        print("[ERROR] Could not find base model layer ('mobilenetv2'). Skipping fine-tuning phase.")
        checkpoint_path_fine = checkpoint_path_init

    # --- Plot Training History ---
    print("[INFO] Plotting training history...")
    # <<< CHANGE: Pass updated save path >>>
    plot_history(history_init, history_fine, save_path='training_history_multi_class.png')

    # --- Evaluation on Test Set ---
    # <<< CHANGE: Updated print statement >>>
    print("\n[INFO] Evaluating final multi-class model on the test set...")

    # Determine the path of the best model
    best_model_path = checkpoint_path_init
    if history_fine and os.path.exists(checkpoint_path_fine):
        print(f"[INFO] Using best model from fine-tuning phase: '{checkpoint_path_fine}'")
        best_model_path = checkpoint_path_fine
    elif os.path.exists(checkpoint_path_init):
         print(f"[INFO] Fine-tuning skipped or checkpoint not found. Using best model from initial phase: '{checkpoint_path_init}'")
    else:
        print(f"[ERROR] No best model checkpoint found at '{checkpoint_path_init}' or '{checkpoint_path_fine}'.")
        print("[INFO] Evaluating model from the end of the last training phase (may be suboptimal).")
        best_model_path = None

    # Load the selected best model
    if best_model_path and os.path.exists(best_model_path):
        print(f"[INFO] Loading model for evaluation from: {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
        print("[INFO] Model loaded successfully for evaluation.")
    elif not best_model_path:
         print("[INFO] Evaluating the model currently in memory.")
    else:
        print(f"[ERROR] Failed to load the model from {best_model_path}. Aborting evaluation.")
        exit(1)

    # Perform Evaluation using the test generator
    print("[INFO] Evaluating model performance on the test dataset...")
    # The evaluate function returns loss and metric values based on model.compile()
    test_results = model.evaluate(
        test_generator,
        verbose=1,
        return_dict=True # Easier to access metrics by name
    )

    print("\n--- Test Set Evaluation Results ---")
    print(f"  Test Loss: {test_results['loss']:.4f}")
    # Access the accuracy metric by the name given ('accuracy')
    if 'accuracy' in test_results:
        print(f"  Test Accuracy: {test_results['accuracy']*100:.2f}%")
    else:
         # Fallback if name isn't found (e.g. default name)
         print(f"  Test Metrics: {test_results}")


    # --- Classification Report ---
    # <<< CHANGE: Updated print statement >>>
    print("\n[INFO] Generating Multi-Class Classification Report for Test Set...")
    # Predict probabilities on the test set
    test_generator.reset()
    steps = int(np.ceil(num_test_samples / BATCH_SIZE))
    print(f"[INFO] Making predictions on {num_test_samples} test samples using {steps} steps...")

    predictions_prob = model.predict(test_generator, steps=steps, verbose=1)

    # Get true labels (integer indices 0, 1, 2...)
    y_true = test_generator.classes
    y_true = y_true[:num_test_samples] # Ensure correct length

    # <<< CHANGE: Get predicted labels from probabilities using argmax >>>
    # Find the index of the highest probability for each sample
    y_pred = np.argmax(predictions_prob, axis=1)

    # Ensure correct length for predictions as well
    if len(y_pred) != num_test_samples:
        print(f"[WARNING] Number of predictions ({len(y_pred)}) does not match number of test samples ({num_test_samples}). Adjusting prediction array.")
        y_pred = y_pred[:num_test_samples]

    print("\n--- Classification Report ---")
    # Use the class names derived from the generator's class_indices
    try:
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names, # Should be ['Fusarium Head Blight', 'Healthy', 'Leaf Blight'] or similar order
            zero_division=0
        )
        print(report)
    except ValueError as e:
        print(f"[ERROR] Could not generate classification report: {e}")
        print("Ensure y_true and y_pred contain valid integer labels and target_names match the number of classes.")
        print(f"y_true unique values: {np.unique(y_true)}")
        print(f"y_pred unique values: {np.unique(y_pred)}")
        print(f"target_names: {class_names}")


    # --- Save the Final Best Model ---
    # <<< CHANGE: Updated final model save path >>>
    final_model_save_path = "final_wheat_multi_class_model.keras"
    if best_model_path and os.path.exists(best_model_path):
        print(f"[INFO] Saving the final best evaluated model from {best_model_path} to: {final_model_save_path}")
        # Reload just to be sure we save the correct one
        model_to_save = tf.keras.models.load_model(best_model_path)
        model_to_save.save(final_model_save_path)
        print("[INFO] Final model saved successfully.")
    elif not best_model_path:
         print(f"[INFO] Saving the model currently in memory to: {final_model_save_path}")
         model.save(final_model_save_path)
         print("[INFO] Final model saved successfully.")
    else:
         print("[WARNING] Could not save final model as the best checkpoint path was invalid.")

    print(f"\n[INFO] Script finished execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
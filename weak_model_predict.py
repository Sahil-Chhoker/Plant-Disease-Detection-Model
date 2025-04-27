import tensorflow as tf
import numpy as np
from PIL import Image
import time
import math

# --- Configuration ---
MODEL_PATH = 'weak_model.tflite'  
IMAGE_PATH = 'input/img5.jpeg'
LABEL_PATH = 'dict.txt'      

# --- Function to Load Labels ---
def load_labels(filename):
    """Loads labels from a text file."""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Label file not found at {filename}")
        return None

# --- Function for Image Preprocessing (Matching Dart Code) ---
def preprocess_image(image_path, input_shape):
    """Loads and preprocesses image like the Dart code.

    Args:
        image_path: Path to the input image.
        input_shape: The shape expected by the model [batch, height, width, channels]

    Returns:
        Preprocessed image data as a NumPy array, or None if error.
    """
    try:
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels

        # 1. Center Crop
        width, height = img.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2
        img_cropped = img.crop((left, top, right, bottom))

        # 2. Resize using Nearest Neighbor
        target_height = input_shape[1]
        target_width = input_shape[2]
        # Ensure PIL version supports Resampling or use older constants
        try:
            resample_method = Image.Resampling.NEAREST
        except AttributeError:
            resample_method = Image.NEAREST # Older PIL versions
        img_resized = img_cropped.resize((target_width, target_height), resample_method)

        # 3. Normalize to [0.0, 1.0] (Matches NormalizeOp(0, 1))
        input_data = np.array(img_resized, dtype=np.float32) / 255.0

        # 4. Add Batch Dimension
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- Main Execution ---

# 1. Load Labels
print(f"Loading labels from: {LABEL_PATH}")
class_labels = load_labels(LABEL_PATH)
if class_labels is None:
    exit()
print(f"Loaded {len(class_labels)} labels: {class_labels}")

# 2. Load the TFLite Model and Allocate Tensors
print(f"\nLoading model from: {MODEL_PATH}")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model or allocating tensors: {e}")
    exit()

# 3. Get Input and Output Details
input_details = interpreter.get_input_details()[0] # Assuming one input
output_details = interpreter.get_output_details()[0] # Assuming one output

input_shape = input_details['shape']
output_shape = output_details['shape']
output_dtype = output_details['dtype'] # Important: Check if float32 or uint8

print("\n--- Model Input Details ---")
print(f"  Shape: {input_shape}")
print(f"  Type: {input_details['dtype']}")
print("--- Model Output Details ---")
print(f"  Shape: {output_shape}")
print(f"  Type: {output_dtype}")

# Verify label count matches output shape
num_expected_classes = output_shape[-1] # Last dimension is usually number of classes
if len(class_labels) != num_expected_classes:
    print(f"\nWarning: Number of labels loaded ({len(class_labels)}) does not match model output size ({num_expected_classes}).")
    # Decide how to handle this - exit or proceed with caution
    # exit()

# 4. Load and Preprocess the Image
print(f"\nLoading and preprocessing image: {IMAGE_PATH}")
input_data = preprocess_image(IMAGE_PATH, input_shape)
if input_data is None:
    exit()

# Verify input data type matches model expectation
if input_data.dtype != input_details['dtype']:
    print(f"\nWarning: Preprocessed image data type ({input_data.dtype}) does not match model input type ({input_details['dtype']}). Attempting to cast.")
    # Be careful with casting, especially if model expects uint8 and you provide float
    try:
        input_data = input_data.astype(input_details['dtype'])
    except Exception as e:
        print(f"Error casting input data type: {e}")
        exit()


# 5. Perform Inference
print("\n--- Running Inference ---")
interpreter.set_tensor(input_details['index'], input_data)

start_time = time.time()
interpreter.invoke()
end_time = time.time()

# Get the raw output tensor
raw_output_data = interpreter.get_tensor(output_details['index']) # Shape is usually (1, num_classes)

inference_time_ms = (end_time - start_time) * 1000
print(f"Inference time: {inference_time_ms:.2f} ms")

# 6. Post-process the Output (Matching Dart Code)
print("\n--- Prediction Results ---")
# Apply the postProcessNormalizeOp (divide by 255.0)
# This is unusual if output_dtype is float32, but matches the Dart code's logic.
# If output_dtype is uint8, this scales it to [0.0, 1.0].
processed_output_data = raw_output_data.astype(np.float32) / 255.0
print(f"Output type was {output_dtype}, applied division by 255 as per Dart code.")

# Get the scores for the single image (remove batch dimension)
scores = processed_output_data[0]

# Find the index of the highest score
predicted_index = np.argmax(scores)
confidence = scores[predicted_index]

# Map the index to the label
if predicted_index < len(class_labels):
    predicted_label = class_labels[predicted_index]
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence (after /255): {confidence:.4f}") # Note: Confidence is after the /255 step
    # print(f"All Scores (after /255): {scores}") # Optional
else:
    print(f"Error: Predicted index {predicted_index} is out of bounds for the loaded labels (count {len(class_labels)}).")
    print("Please check your {LABEL_PATH} file.")

# Example: Print raw output values before division for comparison
# print(f"Raw output values: {raw_output_data[0]}")
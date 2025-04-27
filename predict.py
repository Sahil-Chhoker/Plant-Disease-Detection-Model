import numpy as np
import cv2
import pickle
from tensorflow import keras
import os
import time 

MODEL_PATH = 'plant_species_disease_model_final.keras'
SPECIES_BINARIZER_PATH = 'binarizers/species_binarizer.pkl'
DISEASE_BINARIZER_PATH = 'binarizers/disease_binarizer.pkl'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def preprocess_image(image_path):
    """Loads and preprocesses a single image for prediction."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at path: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cv2.imread failed to load image (may be corrupted or invalid format): {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
        img_array = np.array(img, dtype="float32")
        img_array /= 255.0

        # Add batch dimension (Model expects input shape: (batch_size, height, width, channels))
        img_batch = np.expand_dims(img_array, axis=0) # Shape becomes (1, 224, 224, 3)
        # print(f"[DEBUG] Preprocessed image shape: {img_batch.shape}")
        return img_batch
    except FileNotFoundError as fnf_error:
        print(f"[ERROR] {fnf_error}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to preprocess image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_leaf(image_path):
    """Predicts species and disease for a single leaf image."""

    # Load Model and Binarizers
    try:
        print(f"[INFO] Loading model from: {os.path.abspath(MODEL_PATH)}")
        if not os.path.exists(MODEL_PATH): raise FileNotFoundError(MODEL_PATH)
        model = keras.models.load_model(MODEL_PATH)

        print(f"[INFO] Loading species binarizer from: {os.path.abspath(SPECIES_BINARIZER_PATH)}")
        if not os.path.exists(SPECIES_BINARIZER_PATH): raise FileNotFoundError(SPECIES_BINARIZER_PATH)
        with open(SPECIES_BINARIZER_PATH, 'rb') as f:
            species_binarizer = pickle.load(f)

        print(f"[INFO] Loading disease binarizer from: {os.path.abspath(DISEASE_BINARIZER_PATH)}")
        if not os.path.exists(DISEASE_BINARIZER_PATH): raise FileNotFoundError(DISEASE_BINARIZER_PATH)
        with open(DISEASE_BINARIZER_PATH, 'rb') as f:
            disease_binarizer = pickle.load(f)

    except FileNotFoundError as e:
        print(f"[ERROR] Required file not found: {e}. Make sure model and .pkl files are in the correct relative paths.")
        return None, None, None, None
    except Exception as e:
        print(f"[ERROR] Failed to load model or binarizers: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    # Preprocess the New Image
    print(f"[INFO] Preprocessing image: {image_path}")
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return None, None, None, None

    # Predict
    print("[INFO] Making prediction...")
    start_time = time.time()
    try:
        predictions = model.predict(preprocessed_image)
        # predictions is a list [species_probs_array, disease_probs_array]
        # For a batch size of 1, these arrays might have shape (1, num_classes)
        species_prediction_probs = predictions[0]
        disease_prediction_probs = predictions[1]
        # print(f"[DEBUG] Raw species prediction shape: {species_prediction_probs.shape}")
        # print(f"[DEBUG] Raw disease prediction shape: {disease_prediction_probs.shape}")

        # Decode Predictions
        # Access the probabilities for the first (and only) image in the batch
        species_probs_single = species_prediction_probs[0] # Shape becomes (num_species,)
        disease_probs_single = disease_prediction_probs[0] # Shape becomes (num_disease,)

        # Get the index of the highest probability class from the 1D array
        species_pred_index = np.argmax(species_probs_single) # No axis needed for 1D array
        disease_pred_index = np.argmax(disease_probs_single) # No axis needed for 1D array

        # Get the confidence scores (probabilities) for the predicted classes
        species_confidence = species_probs_single[species_pred_index]
        disease_confidence = disease_probs_single[disease_pred_index]

        # Use the .classes_ attribute of the binarizer to map the index back to the label name
        predicted_species = species_binarizer.classes_[species_pred_index]
        predicted_disease = disease_binarizer.classes_[disease_pred_index]

        end_time = time.time()
        print(f"[INFO] Prediction complete in {end_time - start_time:.3f} seconds.")
        return predicted_species, species_confidence, predicted_disease, disease_confidence

    except IndexError as ie:
        print(f"[ERROR] Index error during prediction decoding: {ie}. This might happen if prediction output format is unexpected or binarizer classes are empty.")
        print(f"[DEBUG] Prediction output structure: {predictions}")
        print(f"[DEBUG] Species Binarizer Classes: {species_binarizer.classes_}")
        print(f"[DEBUG] Disease Binarizer Classes: {disease_binarizer.classes_}")
        return None, None, None, None
    except Exception as e:
        print(f"[ERROR] Failed during prediction or decoding: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    new_image_file = 'input/img.jpeg'

    print(f"--- Running Prediction for Image: {new_image_file} ---")

    if not os.path.exists(new_image_file):
        print(f"[ERROR] Image file not found: {os.path.abspath(new_image_file)}")
    else:
        species, species_conf, disease, disease_conf = predict_leaf(new_image_file)

        # Display results
        if species is not None and disease is not None:
            print("\n--- Prediction Results ---")
            print(f"Predicted Species: {species} (Confidence: {species_conf:.2%})")
            print(f"Predicted Disease/Status: {disease} (Confidence: {disease_conf:.2%})")
            print("------------------------")
        else:
            print("\n[INFO] Prediction failed. Check errors above.")
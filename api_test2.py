import os
import shutil
import re
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import dotenv

dotenv.load_dotenv()

API_URL = "https://serverless.roboflow.com"
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "plant-disease-detection-v2-2nclk/1"

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

SEPARATE_NO_DETECTION = True
NO_DETECTION_FOLDER_NAME = "no_disease_detected"

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

# --- Drawing Configuration ---
BOX_COLOR = "red"
TEXT_COLOR = "white"
TEXT_BG_COLOR = "red"
LINE_THICKNESS = 3
FONT_SIZE = 15

def sanitize_foldername(name):
    """Removes or replaces characters potentially invalid for folder names."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    if not name:
        name = "invalid_classname"
    return name

print("Initializing Roboflow client...")
try:
    CLIENT = InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY
    )
    print("Roboflow client initialized successfully.")
except Exception as e:
    print(f"Error initializing Roboflow client: {e}")
    exit()

if not os.path.isdir(INPUT_FOLDER):
    print(f"Error: Input folder '{INPUT_FOLDER}' not found or is not a directory.")
    exit()

if not os.path.exists(OUTPUT_FOLDER):
    print(f"Creating base output folder: '{OUTPUT_FOLDER}'")
    os.makedirs(OUTPUT_FOLDER)
elif not os.path.isdir(OUTPUT_FOLDER):
    print(f"Error: Output path '{OUTPUT_FOLDER}' exists but is not a directory.")
    exit()

if SEPARATE_NO_DETECTION:
    no_detection_path = os.path.join(OUTPUT_FOLDER, NO_DETECTION_FOLDER_NAME)
    if not os.path.exists(no_detection_path):
        print(f"Creating folder for images with no detections: '{no_detection_path}'")
        os.makedirs(no_detection_path)

try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    print(f"Loaded font 'arial.ttf' with size {FONT_SIZE}.")
except IOError:
    print(f"Warning: Font 'arial.ttf' not found. Using default PIL font.")
    font = ImageFont.load_default()

print("-" * 30)
print(f"Processing images from: '{INPUT_FOLDER}'")
print(f"Saving detected diseased images WITH BOUNDING BOXES into subfolders within: '{OUTPUT_FOLDER}'")
if SEPARATE_NO_DETECTION:
    print(f"Original images with no detections will be saved in: '{NO_DETECTION_FOLDER_NAME}'")
print("-" * 30)

# --- Process Each Image in the Input Folder ---
processed_count = 0
saved_annotated_count = 0
no_detection_count = 0
skipped_count = 0
error_count = 0

for filename in os.listdir(INPUT_FOLDER):
    source_image_path = os.path.join(INPUT_FOLDER, filename)

    # Basic file check
    _, file_extension = os.path.splitext(filename)
    if not os.path.isfile(source_image_path) or file_extension.lower() not in VALID_IMAGE_EXTENSIONS:
        skipped_count += 1
        continue

    processed_count += 1
    print(f"\nProcessing ({processed_count}): {filename}...")

    try:
        # Perform inference
        result = CLIENT.infer(source_image_path, model_id=MODEL_ID)

        # Check for predictions
        if 'predictions' in result and result['predictions']:
            # Find the prediction with the highest confidence
            best_prediction = max(result['predictions'], key=lambda p: p['confidence'])

            confidence = best_prediction['confidence']
            class_name = best_prediction['class']
            print(f"  Detected: '{class_name}' (Confidence: {confidence:.2%})")

            # --- Draw Bounding Box ---
            try:
                # Open the original image
                image = Image.open(source_image_path).convert("RGB")
                draw = ImageDraw.Draw(image)

                # Extract coordinates and dimensions from the best prediction
                x_center = best_prediction['x']
                y_center = best_prediction['y']
                width = best_prediction['width']
                height = best_prediction['height']

                # Convert center coords to top-left (x1,y1) and bottom-right (x2,y2)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # Draw the rectangle
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=BOX_COLOR,
                    width=LINE_THICKNESS
                )

                # Prepare text label
                text = f"{class_name}: {confidence:.1%}" # Slightly shorter text format

                # Calculate text size and position
                try:
                    # Use textbbox for more accurate size calculation in newer Pillow versions
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # Fallback for older Pillow versions
                    text_width, text_height = draw.textsize(text, font=font)

                text_x = x1
                text_y = y1 - text_height - 2 # Position above box
                if text_y < 0: # Adjust if it goes off the top
                    text_y = y1 + 2

                # Draw text background and text
                draw.rectangle(
                    [(text_x, text_y), (text_x + text_width, text_y + text_height)],
                    fill=TEXT_BG_COLOR
                )
                draw.text(
                    (text_x, text_y),
                    text,
                    fill=TEXT_COLOR,
                    font=font
                )
                print(f"  Drew bounding box for '{class_name}'.")

            except Exception as draw_err:
                print(f"  ERROR drawing on image '{filename}': {draw_err}")
                error_count += 1
                continue # Skip saving this image if drawing failed

            # --- Save the ANNOTATED image ---
            target_folder_name = sanitize_foldername(class_name)
            target_folder_path = os.path.join(OUTPUT_FOLDER, target_folder_name)
            os.makedirs(target_folder_path, exist_ok=True)
            target_image_path = os.path.join(target_folder_path, filename)

            try:
                # Save the modified image (with the drawing)
                image.save(target_image_path)
                print(f"  Saved annotated image to -> '{target_folder_name}'")
                saved_annotated_count += 1
            except Exception as save_err:
                print(f"  ERROR saving annotated image '{filename}' to '{target_image_path}': {save_err}")
                error_count += 1

        else:
            # No predictions - handle optional saving of original
            print(f"  No disease detected in '{filename}'.")
            no_detection_count += 1
            if SEPARATE_NO_DETECTION:
                try:
                    no_detection_output_path = os.path.join(OUTPUT_FOLDER, NO_DETECTION_FOLDER_NAME)
                    # Note: target_folder_path should already exist here if needed
                    target_image_path = os.path.join(no_detection_output_path, filename)
                    print(f"  Copying original '{filename}' to -> '{NO_DETECTION_FOLDER_NAME}'")
                    shutil.copy2(source_image_path, target_image_path) # Copy original
                except Exception as copy_err:
                     print(f"  ERROR copying original '{filename}' to '{NO_DETECTION_FOLDER_NAME}': {copy_err}")
                     error_count +=1


    except Exception as e:
        # Catch errors during inference for a specific file
        print(f"  ERROR during inference for '{filename}': {e}")
        error_count += 1

# --- Summary ---
print("-" * 30)
print("Processing Complete.")
print(f"Total image files processed: {processed_count}")
print(f"Diseased images identified and saved (with annotations): {saved_annotated_count}")
print(f"Images with no disease detected: {no_detection_count}")
if SEPARATE_NO_DETECTION:
    print(f"  (Originals copied to the '{NO_DETECTION_FOLDER_NAME}' folder)")
print(f"Files skipped (non-image/subfolders): {skipped_count}")
print(f"Errors encountered during processing/drawing/saving: {error_count}")
print(f"\nSorted annotated images can be found in the subfolders within: '{OUTPUT_FOLDER}'")
print("-" * 30)
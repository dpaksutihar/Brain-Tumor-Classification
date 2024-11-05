import os
import json
import cv2
import numpy as np
import tensorflow as tf
import sys

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load the classification model
classification_model = tf.keras.models.load_model(
    './seg_class/model.h5',
    compile=False
)

# Define class names for classification
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Load the segmentation model
segmentation_model = tf.keras.models.load_model(
    './seg_class/my_model.keras',
    custom_objects={
        "accuracy": tf.keras.metrics.MeanIoU(num_classes=4),
        "dice_coef": None,
        "precision": None,
        "sensitivity": None,
        "specificity": None,
        "dice_coef_necrotic": None,
        "dice_coef_edema": None,
        "dice_coef_enhancing": None
    },
    compile=False
)

# Function to preprocess the image for classification
def preprocess_image_classification(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")
    
    img = cv2.resize(img, (128, 128))
    img_array = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# Function to preprocess the image for segmentation
def preprocess_image_segmentation(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {img_path}")

    img = cv2.resize(img, (128, 128))
    img_array = img.astype('float32') / 255.0

    if img_array.shape[2] >= 2:
        processed_img = img_array[:, :, :2]
    else:
        processed_img = np.concatenate((img_array, img_array[:, :, :1]), axis=-1)

    return np.expand_dims(processed_img, axis=0)

# Function to process a single image and return prediction and segmentation result
def process_image(img_path):
    # Classification
    preprocessed_image_class = preprocess_image_classification(img_path)
    predictions_class = classification_model.predict(preprocessed_image_class, verbose=0)
    predicted_index = np.argmax(predictions_class, axis=1)[0]
    predicted_label = class_names[predicted_index]
    confidence = np.max(predictions_class, axis=1)[0]  # Get the highest confidence

    # Segmentation
    preprocessed_image_seg = preprocess_image_segmentation(img_path)
    predictions_seg = segmentation_model.predict(preprocessed_image_seg, verbose=0)
    
    segmentation_mask = np.argmax(predictions_seg[0], axis=-1)

    # Create a color map for visualization
    colors = np.array([[0, 0, 0],  # Class 0: Background
                       [1, 1, 0],  # Class 1: Necrotic
                       [0, 1, 1],  # Class 2: Edema
                       [0, 0, 1]])  # Class 3: Enhancing

    overlay = np.zeros((*segmentation_mask.shape, 3))

    for class_index in range(len(colors)):
        overlay[segmentation_mask == class_index] = colors[class_index]

    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Original image not found or unable to read: {img_path}")

    original_img_resized = cv2.resize(original_img, (128, 128))
    combined_image = (original_img_resized.astype(float) / 255 + overlay * 0.4)

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, (combined_image * 255).astype(np.uint8))

    return predicted_label, confidence, output_path

# Main function to process multiple images and return the one with the highest percentage
def main(image_paths):
    best_result = None
    highest_confidence = -1
    
    for img_path in image_paths:
        try:
            predicted_label, confidence, output_path = process_image(img_path)
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_result = {
                    'image_path': img_path,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'segmented_image_path': output_path
                }
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    return best_result

if __name__ == "__main__":
    # Expecting multiple image paths passed as arguments
    img_paths = sys.argv[1:]  # List of image paths passed as command-line arguments
    if not img_paths:
        raise ValueError("Please provide image paths as arguments.")

    best_response = main(img_paths)

    # Output the JSON response for the image with the highest percentage
    print(json.dumps(best_response, indent=2))

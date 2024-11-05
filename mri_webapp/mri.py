import os
import json
import cv2
import numpy as np
import tensorflow as tf
import sys

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all messages except ERROR
tf.get_logger().setLevel('ERROR')  # Set logger level to ERROR

# Load the classification model
classification_model = tf.keras.models.load_model(
    './seg_class/model.h5',
    compile=False  # Avoid unnecessary warnings about metrics if just loading for inference
)

# Define class names for classification
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Load your segmentation model
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

# Main function to process the image
def process_image(img_path):
    # Classification
    preprocessed_image_class = preprocess_image_classification(img_path)
    predictions_class = classification_model.predict(preprocessed_image_class, verbose=0)
    
    # Convert predictions to a dictionary with class labels and their probabilities
    class_probabilities = {class_names[i]: float(predictions_class[0][i]) for i in range(len(class_names))}
    
    predicted_index = np.argmax(predictions_class, axis=1)[0]
    predicted_label = class_names[predicted_index]

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

    return predicted_label, class_probabilities, output_path

def main(img_path):
    try:
        predicted_label, class_probabilities, output_path = process_image(img_path)
        result = {
            'predicted_label': class_names,
            'class_probabilities': class_probabilities,
            'segmented_image_path': output_path,
        }
    except Exception as e:
        result = {
            "error": str(e),
            "message": "Error processing image."
        }

    return result  # Return the result without formatting as JSON here

if __name__ == "__main__":
    img_path = sys.argv[1]
    response = main(img_path)

    # Output the JSON response without extra formatting
    print(json.dumps(response, indent=4))  # Output the JSON response formatted for readability

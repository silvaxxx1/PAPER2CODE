import tensorflow as tf
from model.ResNet50 import ResNetModel
import numpy as np
import json
import sys

def load_config(config_file):
    """
    Load the configuration from the specified JSON file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: Loaded configuration.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_model(model_file):
    """
    Load the trained ResNet model from the specified HDF5 file.

    Args:
        model_file (str): Path to the HDF5 model file.

    Returns:
        tf.keras.Model: Loaded model.
    """
    model = tf.keras.models.load_model(model_file)
    return model

def preprocess_image(image_path, image_size):
    """
    Preprocess the input image.

    Args:
        image_path (str): Path to the input image file.
        image_size (int): The desired size of the image after resizing.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_size, image_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values to [0, 1]
    return img_array

def predict(model, image_path, image_size, class_names):
    """
    Make predictions using the provided model on the input image.

    Args:
        model (tf.keras.Model): Trained model for prediction.
        image_path (str): Path to the input image file.
        image_size (int): The desired size of the image after resizing.
        class_names (list): List of class names for mapping predicted class indices.

    Returns:
        str: Predicted class label.
    """
    # Preprocess the input image
    img_array = preprocess_image(image_path, image_size)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_names[predicted_class_index]
    return predicted_class_label

if __name__ == "__main__":
    # Load configuration
    config = load_config("config.json")

    # Load model
    model = load_model("resnet_model.h5")

    # Load class names
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # Example class names for CIFAR-10

    # Make predictions
    image_path = sys.argv[1]  # Get image path from command line argument
    predicted_class_label = predict(model, image_path, config["image_size"], class_names)

    print("Predicted Class:", predicted_class_label)

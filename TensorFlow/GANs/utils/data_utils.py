import tensorflow as tf 
import os 
import zipfile 
import glob
import numpy as np
from tqdm import tqdm
from urllib.request import urlretrieve
from PIL import Image as Im

def get_data():
    """
    Downloads and extracts a dataset of images from a specified URL.

    This function performs the following steps:
    1. Creates a directory at '/data/celeb_faces' if it does not already exist.
    2. Downloads the dataset from the specified URL.
    3. Extracts the downloaded ZIP file into the created directory.

    Args:
        None

    Returns:
        None
    """
    try:
        # Create directory if it does not exist
        os.mkdir('/data/celeb_faces')
    except OSError:
        # Directory already exists, no need to create
        pass 
    
    data_url = "https://storage.googleapis.com/learning-datasets/Resources/archive.zip"
    data_file = "archive.zip"
    download_path = '/data/celeb_faces'
    
    # Download the dataset
    urlretrieve(data_url, data_file)
    
    # Extract the ZIP file
    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(download_path)
        zip_ref.close()

    print("Download and extraction complete")

def load_data(batch_size, resize=64, crop_size=128):
    """
    Loads and preprocesses images from the dataset.

    This function performs the following steps:
    1. Loads all image file paths from '/data/celeb_faces'.
    2. Crops and resizes the images.
    3. Splits the images into two halves.
    4. Converts the images to a TensorFlow dataset with batching and prefetching.

    Args:
        batch_size (int): The batch size for the TensorFlow dataset.
        resize (int): The size to resize images to (height and width).
        crop_size (int): The size to crop images to (height and width).

    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for training.
    """
    # Get sorted list of image paths
    images_path = sorted(glob.glob("/data/celeb_faces/*.jpg"))
    
    # Initialize an array to hold the images
    images = np.zeros((len(images_path), resize, resize, 3), np.uint8)
    print("Loading images")
    
    # Load and preprocess each image
    for i, path in tqdm(enumerate(images_path)):
        with Im.open(path) as img:
            # Crop the image to the desired crop size
            left = (img.size[0] - crop_size) // 2
            top = (img.size[1] - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            img = img.crop((left, top, right, bottom))
            
            # Resize the image
            img = img.resize((resize, resize), Im.LANCZOS)
            images[i] = np.asarray(img, np.uint8)
    
    # Split the images into two halves
    split = images.shape[0] // 2
    images1 = images[:split] 
    images2 = images[split:]
    del images  # Free up memory by deleting the original images array
    
    def preprocess(x):
        """
        Normalizes image data to the range [-1, 1].

        Args:
            x (tf.Tensor): A tensor of image data.

        Returns:
            tf.Tensor: The normalized image data.
        """
        x = tf.cast(x, tf.float32) / 127.5 - 1.0
        return x
    
    # Create a TensorFlow dataset
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((images1, images2))
    dataset = dataset.map(lambda x1, x2: (preprocess(x1), preprocess(x2)))
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTO)
    
    return dataset

# # Use the function to get the strategy and load the dataset
# strategy = get_strategy()
# batch_size = 8
# batch_size = batch_size * strategy.num_replicas_in_sync  # Adjust batch size for distributed strategy
# dataset = load_data(batch_size)
# out_dir = "celeba_out"

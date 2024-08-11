import tensorflow as tf
import zipfile
import os

# Define class names for segmentation
class_names = [
    'sky', 'building', 'column/pole', 'road', 'side walk', 'vegetation', 
    'traffic light', 'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void'
]

def file_to_image_mask(img_file, annotation_file, h=224, w=224):
    """
    Processes image and annotation files for training a segmentation model.
    """
    # Read image and annotation files
    img_raw = tf.io.read_file(img_file)
    annotation_raw = tf.io.read_file(annotation_file)
    
    # Decode images and annotations
    image = tf.image.decode_jpeg(img_raw, channels=3)
    annotation = tf.image.decode_jpeg(annotation_raw, channels=1)
    
    # Resize images and annotations
    image = tf.image.resize(image, (h, w))
    annotation = tf.image.resize(annotation, (h, w))
    
    # Reshape and normalize image
    image = tf.reshape(image, (h, w, 3))
    image = image / 127.5 - 1
    
    # Process annotation into one-hot encoded format
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (h, w, 1))
    
    stack = []
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:, :, 0], c)
        stack.append(tf.cast(mask, dtype=tf.int32))
    
    annotation = tf.stack(stack, axis=2)
    
    # Debug prints
    #print(f"Processed image shape: {image.shape}")
    #print(f"Processed annotation shape: {annotation.shape}")
    
    return image, annotation

def get_dataset_slice_paths(img_dir, label_dir):
    """
    Retrieves file paths for images and their corresponding annotations.

    Args:
        img_dir (str): Directory containing image files.
        label_dir (str): Directory containing annotation files.

    Returns:
        Tuple[List[str], List[str]]: Lists of image file paths and annotation file paths.
    """
    image_file_list = sorted(os.listdir(img_dir))
    label_file_list = sorted(os.listdir(label_dir))
    
    img_path = [os.path.join(img_dir, file_name) for file_name in image_file_list]
    label_map_path = [os.path.join(label_dir, file_name) for file_name in label_file_list]
    
    return img_path, label_map_path

BATCH_SIZE = 64
AUTO = tf.data.experimental.AUTOTUNE

def get_training_dataset(img_path, label_map_path):
    """
    Creates a TensorFlow Dataset for training.

    Args:
        img_path (List[str]): List of image file paths.
        label_map_path (List[str]): List of annotation file paths.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object for training.
    """
    training_dataset = tf.data.Dataset.from_tensor_slices((img_path, label_map_path))
    training_dataset = training_dataset.map(file_to_image_mask, num_parallel_calls=AUTO)
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(AUTO)
    
    return training_dataset

def get_validation_dataset(img_path, label_map_path):
    """
    Creates a TensorFlow Dataset for validation.

    Args:
        img_path (List[str]): List of image file paths.
        label_map_path (List[str]): List of annotation file paths.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object for validation.
    """
    validation_dataset = tf.data.Dataset.from_tensor_slices((img_path, label_map_path))
    validation_dataset = validation_dataset.map(file_to_image_mask, num_parallel_calls=AUTO)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    
    return validation_dataset

# Get the paths to the images
training_image_paths, training_label_map_paths = get_dataset_slice_paths(
    r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\annotations_prepped_train',
    r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\images_prepped_train'
)
validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(
    r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\annotations_prepped_test',
    r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\images_prepped_test'
)

# Debug prints to check if file counts match
#print(f'Number of training images: {len(training_image_paths)}')
#print(f'Number of training annotations: {len(training_label_map_paths)}')
#print(f'Number of validation images: {len(validation_image_paths)}')
#print(f'Number of validation annotations: {len(validation_label_map_paths)}')

# Ensure the counts match before creating datasets
assert len(training_image_paths) == len(training_label_map_paths), "Training image and annotation counts do not match!"
assert len(validation_image_paths) == len(validation_label_map_paths), "Validation image and annotation counts do not match!"

# Generate the train and validation sets
training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

# Check dataset output
#for image, annotation in training_dataset.take(1):
    #print("Image shape:", image.shape)
    #print("Annotation shape:", annotation.shape)
    

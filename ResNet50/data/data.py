import tensorflow as tf
from keras.datasets import cifar10 
from keras.utils import to_categorical

def load_data(image_size, batch_size):
    """
    Load and preprocess the CIFAR-10 dataset.

    Args:
        image_size (int): The desired size of the images after resizing.
        batch_size (int): The batch size for training and validation.

    Returns:
        train_dataset: Preprocessed training dataset.
        test_dataset: Preprocessed validation dataset.
    """
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize pixel values to the range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0 

    # Resize images to the desired size
    X_train_resized = tf.image.resize(X_train, (image_size, image_size))
    X_test_resized = tf.image.resize(X_test, (image_size, image_size))

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10  # Number of classes in CIFAR-10 dataset
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_resized, y_train)).shuffle(10000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_resized, y_test)).batch(batch_size)

    return train_dataset, test_dataset

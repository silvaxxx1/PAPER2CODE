import tensorflow as tf
import tensorflow_datasets as tfds


BATCH_SIZE = 64
BUFFER_SIZE = 1000

def get_dataset():
    dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
    return dataset, info

dataset , info = get_dataset()

def flip(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

def norm(image, mask):
    image = tf.cast(image, dtype=tf.float32)
    mask -= 1
    return image, mask

@tf.function
def load_train(datapoint):
    image = tf.image.resize(datapoint['image'], (128,128), method='nearest')
    mask = tf.image.resize(datapoint['segmentation_mask'], (128,128), method='nearest')
    image, mask = flip(image, mask)
    image, mask = norm(image, mask)
    return image, mask

def load_test(datapoint):
    image = tf.image.resize(datapoint['image'], (128,128), method='nearest')
    mask = tf.image.resize(datapoint['segmentation_mask'], (128,128), method='nearest')
    image, mask = norm(image, mask)
    return image, mask

def training_dataset():
    # Dataset preparation
    AUTO = tf.data.experimental.AUTOTUNE
    train = dataset['train'].map(load_train, num_parallel_calls=AUTO)
    

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(AUTO)
    
    return train_dataset 

def test_dataset():
    test = dataset['test'].map(load_test)
    test_dataset = test.batch(BATCH_SIZE)
    return test_dataset 




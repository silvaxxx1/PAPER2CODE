import tensorflow as tf 
import numpy as np 
from data import get_dataset , training_dataset , test_dataset 
from Model import unet


dataset , info = get_dataset()
dataset , info = get_dataset() 
train_dataset =  training_dataset()
test_dataset = test_dataset()
model = unet()

BATCH_SIZE = 64
BUFFER_SIZE = 1000

# Prediction Utilities

def get_test_image_and_annotation_arrays():
    ''' Unpacks the test dataset and returns the input images and segmentation masks '''

    ds = test_dataset.unbatch()
    ds = ds.batch(info.splits['test'].num_examples)
    
    images = []
    y_true_segments = []

    for image, annotation in ds.take(1):
        y_true_segments = annotation.numpy()
        images = image.numpy()
    
    y_true_segments = y_true_segments[:(info.splits['test'].num_examples - (info.splits['test'].num_examples % BATCH_SIZE))]
    
    return images[:(info.splits['test'].num_examples - (info.splits['test'].num_examples % BATCH_SIZE))], y_true_segments


def create_mask(pred_mask):
    '''
    Creates the segmentation mask by getting the channel with the highest probability. Remember that we
    have 3 channels in the output of the UNet. For each pixel, the predicition will be the channel with the
    highest probability.
    '''
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0].numpy()


def make_predictions(model,image, mask, num=1):
    ''' Feeds an image to a model and returns the predicted mask. '''

    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    pred_mask = model.predict(image)
    pred_mask = create_mask(pred_mask)

    return pred_mask 
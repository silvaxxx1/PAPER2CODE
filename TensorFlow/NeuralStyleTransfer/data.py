# the data utils for the project
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    tensor_shape = tensor.shape
    if len(tensor_shape) > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)
 


def load_image(img_path):
    max_dim = 512
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    shape = tf.shape(image)[:-1]
    shape = tf.cast(shape,tf.float32)
    longest_dim = max(shape)
    scale = max_dim / longest_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image,tf.uint8) 
    
    return image 

def load_style_and_content(style_path, content_path):
    style_image = load_image(style_path)
    content_image = load_image(content_path)
    return style_image, content_image

def clip_image(image, min=0.0, max=255.0):
    image = tf.clip_by_value(image, clip_value_min=min, clip_value_max=max)

def perpocess_image(image):
    image = tf.cast(image, tf.float32)
    image  = (image / 127.5) - 1
    
    return image

def imshow(image, title=None):
  '''displays an image with a corresponding title'''
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
    
    
def show_images_with_objects(images, titles=[]):
  '''displays a row of images with corresponding titles'''
  if len(images) != len(titles):
    return

  plt.figure(figsize=(20, 12))
  for idx, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(1, len(images), idx + 1)
    plt.xticks([])
    plt.yticks([])
    imshow(image, title)



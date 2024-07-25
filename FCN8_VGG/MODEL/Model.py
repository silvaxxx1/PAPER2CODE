import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input
from keras import Model
from load_weights import get_wieghts
import os

# URL of the VGG16 weights
weights_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_file = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Download the weights file if it does not exist
if not os.path.exists(weights_file):
    get_wieghts(weights_url, weights_file)

def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    """
    Define a block of Conv2D layers followed by a MaxPooling layer.

    Args:
    x: Input tensor.
    n_convs: Number of Conv2D layers in the block.
    filters: Number of filters for Conv2D layers.
    kernel_size: Size of the Conv2D kernel.
    activation: Activation function for Conv2D layers.
    pool_size: Size of the MaxPooling window.
    pool_stride: Stride for MaxPooling.
    block_name: Name prefix for layers in this block.

    Returns:
    x: Output tensor after applying Conv2D layers and MaxPooling.
    """
    for i in range(n_convs):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation=activation,
                   padding='same',
                   name=f"{block_name}_conv{i+1}")(x)
        
    x = MaxPool2D(pool_size=pool_size,
                  strides=pool_stride,
                  name=f"{block_name}_pool")(x)
    
    return x

def VGG16(input_image):
    """
    Build the VGG16 architecture up to the last convolutional layer.

    Args:
    input_image: Input tensor with shape (224, 224, 3).

    Returns:
    Tuple of tensors from different layers to be used in the decoder.
    """
    x = block(input_image,
              n_convs=2,
              filters=64,
              kernel_size=(3,3),
              activation='relu',
              pool_size=(2,2),
              pool_stride=(2,2),
              block_name="block1")
    p1 = x 
    
    x = block(x,
              n_convs=2,
              filters=128,
              kernel_size=(3,3),
              activation='relu',
              pool_size=(2,2),
              pool_stride=(2,2),
              block_name="block2")
    p2 = x 
    
    x = block(x,
              n_convs=3,
              filters=256,
              kernel_size=(3,3),
              activation='relu',
              pool_size=(2,2),
              pool_stride=(2,2),
              block_name="block3")
    p3 = x 
    
    x = block(x,
              n_convs=3,
              filters=512,
              kernel_size=(3,3),
              activation='relu',
              pool_size=(2,2),
              pool_stride=(2,2),
              block_name="block4")
    p4 = x 
    
    x = block(x,
              n_convs=3,
              filters=512,
              kernel_size=(3,3),
              activation='relu',
              pool_size=(2,2),
              pool_stride=(2,2),
              block_name="block5")
    p5 = x 
    
    # Create the VGG16 model with the specified weights
    vgg16_model = Model(inputs=input_image, outputs=p5)
    vgg16_model.load_weights(weights_file)
    
    n = 4096
    c6 = tf.keras.layers.Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6")(p5)
    c7 = tf.keras.layers.Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7")(c6)
    
    return (p1, p2, p3, p4, c7)

def decoder(convs, n_classes):
    """
    Build the decoder part of the U-Net architecture.

    Args:
    convs: Tuple of tensors from different layers of the encoder.
    n_classes: Number of output classes for segmentation.

    Returns:
    Output tensor with shape (224, 224, n_classes) representing the segmentation map.
    """
    f1, f2, f3, f4, f5 = convs
    
    # Decoder path
    o = Conv2DTranspose(n_classes,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        use_bias=False)(f5)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)
    
    o2 = Conv2D(n_classes,
                (1, 1),
                activation='relu',
                padding='same')(f4)
    o = tf.keras.layers.Add()([o, o2])
    
    o = Conv2DTranspose(n_classes,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        use_bias=False)(o)
    o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)    
    
    o2 = Conv2D(n_classes,
                (1, 1),
                activation='relu',
                padding='same')(f3)
    o = tf.keras.layers.Add()([o, o2])
    
    o = Conv2DTranspose(n_classes,
                        kernel_size=(8, 8),
                        strides=(8, 8),
                        use_bias=False)(o)

    # Append a softmax activation to get class probabilities
    o = tf.keras.layers.Activation('softmax')(o)

    return o

def seg_model():
    """
    Build the full segmentation model using VGG16 as the encoder and a custom decoder.

    Returns:
    A Keras Model object for segmentation.
    """
    input = Input(shape=(224, 224, 3))
    convs = VGG16(input_image=input)
    output = decoder(convs=convs, n_classes=12)
    model = tf.keras.models.Model(inputs=input, outputs=output)

    return model

# Instantiate the model and print its summary
#model = seg_model()
#model.summary()

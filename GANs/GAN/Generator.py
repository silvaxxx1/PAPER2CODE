import tensorflow as tf
from utils.model_utils import get_norm_layer, get_initializers

def create_generator(input_shape=(1,1,128),
                     upsampling_factor=4,
                     output_channels=3,
                     norm='batch_norm',
                     dim=64,
                     name='generator'):
    """
    Creates a generator model for generating images.

    Args:
        input_shape (tuple): Shape of the input tensor (default is (1, 1, 128)).
        upsampling_factor (int): Number of upsampling layers (default is 4).
        output_channels (int): Number of output channels (default is 3 for RGB images).
        norm (str): Type of normalization layer to use ('batch_norm', 'layer_norm', etc.).
        dim (int): Initial number of filters in the first convolutional layer (default is 64).
        name (str): Name of the model (default is 'generator').

    Returns:
        tf.keras.Model: A Keras model representing the generator.
    """

    # Get the normalization layer and initializers
    Normalization = get_norm_layer(norm)
    conv_initializer, bn_gamma_initializer = get_initializers()
    
    # Define the input layer
    x = inputs = tf.keras.layers.Input(shape=input_shape)
    
    # 1. Define the first Conv2DTranspose layer (upsamples 1x1 -> 4x4)
    # Calculate dimensions for the first layer based on the upsampling factor
    dimensions = min(dim * 2 ** (upsampling_factor - 1), dim * 8)
    x = tf.keras.layers.Conv2DTranspose(dimensions, 4,
                                        strides=2,
                                        padding='valid',
                                        use_bias=False,
                                        # kernel_initializer=conv_initializer
                                        )(x)
    # Apply normalization and ReLU activation
    x = Normalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 2. Define additional Conv2DTranspose layers (e.g., 4x4 -> 8x8 -> 16x16 -> ...)
    for i in range(upsampling_factor - 1):
        # Calculate dimensions for each layer
        dimensions = min(dim * 2 ** (upsampling_factor - 2 - i), dim * 8)
        x = tf.keras.layers.Conv2DTranspose(dimensions, 4,
                                            strides=2,
                                            padding='same',
                                            use_bias=False,
                                            # kernel_initializer=conv_initializer
                                            )(x)
        # Apply normalization and ReLU activation
        x = Normalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    # 3. Define the last Conv2DTranspose layer with tanh activation
    x = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                        strides=2,
                                        padding='same',
                                        # kernel_initializer=conv_initializer
                                        )(x)
    # Apply tanh activation to the output layer
    outputs = tf.keras.layers.Activation('tanh')(x)
    
    # Create and return the Keras model
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


# model = create_generator()
# model.summary()
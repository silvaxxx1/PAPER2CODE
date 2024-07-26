import tensorflow as tf 
 

# Encoder Utilities

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    ''' Adds 2 convolutional layers with the parameters passed to it

    Args:
        input_tensor (tensor) -- the input tensor
        n_filters (int) -- number of filters
        kernel_size (int) -- kernel size for the convolution

    Returns:
        tensor of output features
    '''
    # first layer
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, 
                                   kernel_size=(kernel_size, kernel_size), 
                                   kernel_initializer='he_normal', 
                                   padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
    
    return x


def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
    ''' Adds two convolutional blocks and then perform down sampling on output of convolutions.

    Args:
        input_tensor (tensor) -- the input tensor
        n_filters (int) -- number of filters
        kernel_size (int) -- kernel size for the convolution

    Returns:
        f - the output features of the convolution block 
        p - the maxpooled features with dropout
    '''

    f = conv2d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(f)
    p = tf.keras.layers.Dropout(0.3)(p)

    return f, p


def encoder(inputs):
    ''' This function defines the encoder or downsampling path.

    Args:
        inputs (tensor) -- batch of input images

    Returns:
        p4 - the output maxpooled features of the last encoder block
        (f1, f2, f3, f4) - the output features of all the encoder blocks
    '''
    f1, p1 = encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3)
    f2, p2 = encoder_block(p1, n_filters=128, pool_size=(2,2), dropout=0.3)
    f3, p3 = encoder_block(p2, n_filters=256, pool_size=(2,2), dropout=0.3)
    f4, p4 = encoder_block(p3, n_filters=512, pool_size=(2,2), dropout=0.3)

    return p4, (f1, f2, f3, f4)

def bottleneck(inputs):
  '''
  This function defines the bottleneck convolutions to extract more features before the upsampling layers.
  '''
  
  bottle_neck = conv2d_block(inputs, n_filters=1024)

  return bottle_neck

# Decoder Utilities

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    ''' Defines the one decoder block of the UNet

    Args:
        inputs (tensor) -- batch of input features
        conv_output (tensor) -- features from an encoder block
        n_filters (int) -- number of filters
        kernel_size (int) -- kernel size
        strides (int) -- strides for the deconvolution/upsampling
        padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

    Returns:
        c (tensor) -- output features of the decoder block
    '''
    u = tf.keras.layers.Conv2DTranspose(n_filters, 
                                        kernel_size, 
                                        strides=strides, 
                                        padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)

    return c


def decoder(inputs, convs, output_channels):
    ''' Defines the decoder of the UNet chaining together 4 decoder blocks. 
    
    Args:
        inputs (tensor) -- batch of input features
        convs (tuple) -- features from the encoder blocks
        output_channels (int) -- number of classes in the label map

    Returns:
        outputs (tensor) -- the pixel wise label map of the image
    '''
    
    f1, f2, f3, f4 = convs

    c6 = decoder_block(inputs, f4, 
                       n_filters=512, 
                       kernel_size=(3,3), 
                       strides=(2,2), 
                       dropout=0.3)
    c7 = decoder_block(c6, f3, 
                       n_filters=256, 
                       kernel_size=(3,3), 
                       strides=(2,2), 
                       dropout=0.3)
    c8 = decoder_block(c7, f2, 
                       n_filters=128, 
                       kernel_size=(3,3), 
                       strides=(2,2), 
                       dropout=0.3)
    c9 = decoder_block(c8, f1, 
                       n_filters=64, 
                       kernel_size=(3,3), 
                       strides=(2,2), 
                       dropout=0.3)

    outputs = tf.keras.layers.Conv2D(output_channels, 
                                     kernel_size=(1, 1), 
                                     activation='softmax')(c9)

    return outputs

OUTPUT_CHANNELS = 3

def unet():
    ''' Defines the UNet by connecting the encoder, bottleneck and decoder. '''

    # specify the input shape
    inputs = tf.keras.layers.Input(shape=(128, 128,3,))

    # feed the inputs to the encoder
    encoder_output, convs = encoder(inputs)

    # feed the encoder output to the bottleneck
    bottle_neck = bottleneck(encoder_output)

    # feed the bottleneck and encoder block outputs to the decoder
    # specify the number of classes via the `output_channels` argument
    outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)
    
    # create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# instantiate the model
model = unet()

# see the resulting model architecture
model.summary()
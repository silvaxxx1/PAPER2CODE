import tensorflow as tf
from utils.model_utils import get_norm_layer, get_initializers

def create_discriminator(input_shape=(64,64,3),
                         dim=64,
                         n_downsampling=4,
                         norm='batch_norm',
                         name='discriminator'):
    Normalization = get_norm_layer(norm)
    connv_initializer, bn_gamma_initializer = get_initializers()

    # 0. Define the input layer
    x = inputs = tf.keras.layers.Input(shape=input_shape)

    # 1. Define the first Conv2D layer
    x = tf.keras.layers.Conv2D(dim,4,
                               strides=2,
                               padding='same',
                               #kernel_initializer=connv_initializer
                               )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # 2. Define the remaining Conv2D layers
    for i in range(n_downsampling - 1):
        dimension = min(dim *2**(i+1), dim * 8)
        x = tf.keras.layers.Conv2D(dimension,4,
                                   strides=2,
                                   padding='same',
                                   use_bias=False,
                                   #kernel_initializer=connv_initializer
                                   )(x)
        x = Normalization(
                          #gamma_initializer=bn_gamma_initializer
                          )(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # 3. Define logits layer
    x = tf.keras.layers.Conv2D(1,4,
                               strides=1,
                               padding='valid',
                               #kernel_initializer=connv_initializer
                               )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


model = create_discriminator()
model.summary()


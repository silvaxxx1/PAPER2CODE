import tensorflow as tf

def get_norm_layer(norm):
    """
    Returns a normalization layer based on the specified norm type.

    Args:
        norm (str): The type of normalization layer to return. Must be one of "NA", "batch_norm", "layer_norm", or "instance_norm".

    Returns:
        Callable: The corresponding normalization layer or a lambda function that returns the input unchanged.

    Raises:
        ValueError: If the norm parameter is not one of the valid options.
    """
    if norm == "NA":
        # No normalization
        return lambda x: x
    elif norm == "batch_norm":
        # Batch normalization
        return tf.keras.layers.BatchNormalization
    elif norm == "layer_norm":
        # Layer normalization
        return tf.keras.layers.LayerNormalization
    elif norm == "instance_norm":
        # Instance normalization (requires custom implementation)
        return tf.keras.layers.LayerNormalization  # Placeholder for actual instance normalization
    else:
        raise ValueError("Invalid normalization layer: {}".format(norm))

def get_initializers():
    """
    Provides initializers for convolutional and dense layers.

    The convolutional initializer is a RandomNormal initializer with a mean of 0.0 and a standard deviation of 0.02.
    The dense initializer is a RandomNormal initializer with a mean of 1.0 and a standard deviation of 0.02.

    Returns:
        Tuple[tf.keras.initializers.Initializer, tf.keras.initializers.Initializer]: A tuple containing:
            - The initializer for convolutional layers.
            - The initializer for dense layers.
    """
    return (tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),  # Initializer for convolutional layers
            tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02))  # Initializer for dense layers

def gradient_penalty(f, real, fake, mode):
    """
    Computes the gradient penalty for a given function `f` based on real and fake data.

    Args:
        f (Callable): The function to compute the gradient penalty for. Typically a discriminator in GANs.
        real (tf.Tensor): The real data tensor.
        fake (tf.Tensor, optional): The fake data tensor. Used only if mode is 'wgan-gp'.
        mode (str): The mode for computing gradient penalty. Must be one of 'none', 'dragan', or 'wgan-gp'.

    Returns:
        tf.Tensor: The gradient penalty tensor.

    Raises:
        ValueError: If the mode is not one of 'none', 'dragan', or 'wgan-gp'.
    """
    def _gradient_penalty(f, real, fake=None):
        """
        Computes gradient penalty for DRAGAN or WGAN-GP.

        Args:
            f (Callable): The function to compute the gradient penalty for.
            real (tf.Tensor): The real data tensor.
            fake (tf.Tensor, optional): The fake data tensor.

        Returns:
            tf.Tensor: The computed gradient penalty.
        """
        def _interpolate(a, b=None):
            """
            Interpolates between two tensors.

            Args:
                a (tf.Tensor): The first tensor.
                b (tf.Tensor, optional): The second tensor. If None, a random interpolation is used.

            Returns:
                tf.Tensor: The interpolated tensor.
            """
            if b is None:  # DRAGAN interpolation
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        # Interpolate between real and fake data
        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        # Compute gradient and its norm
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        # Compute gradient penalty
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        # No gradient penalty
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        # Compute gradient penalty for DRAGAN
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        # Compute gradient penalty for WGAN-GP
        gp = _gradient_penalty(f, real, fake)
    else:
        raise ValueError("Invalid mode for gradient penalty: {}".format(mode))

    return gp

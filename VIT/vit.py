import tensorflow as tf
from keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Input, Embedding, Concatenate
from keras.models import Model

def patch_embedding(inputs, config):
    """
    Patch Embedding function.

    Args:
        inputs (tf.Tensor): Input tensor.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tf.Tensor: Patch embeddings.
    """
    hidden_dim = config["hidden_dim"]
    patch_embed = Dense(hidden_dim)(inputs)
    return patch_embed

def positional_encoding(inputs, config):
    """
    Positional Encoding function.

    Args:
        inputs (tf.Tensor): Input tensor.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tf.Tensor: Positional embeddings.
    """
    num_patches = config["num_patches"]
    hidden_dim = config["hidden_dim"]
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = Embedding(input_dim=num_patches, output_dim=hidden_dim)(positions)
    return pos_embed

class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def transformer_encoder(x, config):
    """
    Transformer Encoder function.

    Args:
        x (tf.Tensor): Input tensor.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tf.Tensor: Encoded tensor.
    """
    num_heads = config["num_heads"]
    hidden_dim = config["hidden_dim"]
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=num_heads, key_dim=hidden_dim
    )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, config)
    x = Add()([x, skip_2])
    return x

def mlp(x, config):
    """
    Multi-Layer Perceptron (MLP) function.

    Args:
        x (tf.Tensor): Input tensor.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tf.Tensor: Output tensor after passing through the MLP layers.
    """
    x = Dense(config["mlp_dim"], activation="gelu")(x)
    x = Dropout(config["dropout_rate"])(x)
    x = Dense(config["hidden_dim"])(x)
    x = Dropout(config["dropout_rate"])(x)
    return x

def assemble_components(inputs, config):
    """
    Assemble Components function.

    Args:
        inputs (tf.Tensor): Input tensor.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tf.Tensor: Output tensor.
    """
    patch_embeddings = patch_embedding(inputs, config)
    positional_embeddings = positional_encoding(inputs, config)
    embeddings = patch_embeddings + positional_embeddings

    token = ClassToken()(embeddings)
    x = Concatenate(axis=1)([token, embeddings])

    for _ in range(config["num_layers"]):
        x = transformer_encoder(x, config)

    x = LayerNormalization()(x)
    x = x[:, 0, :]
    x = Dropout(0.1)(x)
    outputs = Dense(10, activation="softmax")(x)
    return outputs

def vision_transformer(config):
    """
    Vision Transformer function.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tf.keras.Model: Vision Transformer model.
    """
    inputs = Input((config["num_patches"], config["patch_size"] * config["patch_size"] * config["num_channels"]))
    outputs = assemble_components(inputs, config)
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    config = {
        "num_layers": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "num_patches": 256,
        "patch_size": 32,
        "num_channels": 3
    }

    model = vision_transformer(config)
    model.summary()

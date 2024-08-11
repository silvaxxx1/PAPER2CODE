import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model
from .layers import IdentityBlock, ConvBlock

class ResNetModel(Model):
    """
    A custom ResNet model for image classification.

    Args:
        input_shape (tuple): The shape of the input image (height, width, channels).
        num_classes (int): The number of classes for classification.
    """
    def __init__(self, input_shape, num_classes):
        super(ResNetModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """
        Constructs the ResNet model architecture.
        """
        inputs = Input(shape=self.input_shape)

        # Initial layers
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Stage 1
        x = ConvBlock([64, 64, 256], strides=1)(x)
        x = IdentityBlock([64, 64, 256])(x)
        x = IdentityBlock([64, 64, 256])(x)

        # Stage 2
        x = ConvBlock([128, 128, 512], strides=2)(x)
        x = IdentityBlock([128, 128, 512])(x)
        x = IdentityBlock([128, 128, 512])(x)
        x = IdentityBlock([128, 128, 512])(x)

        # Stage 3
        x = ConvBlock([256, 256, 1024], strides=2)(x)
        x = IdentityBlock([256, 256, 1024])(x)
        x = IdentityBlock([256, 256, 1024])(x)
        x = IdentityBlock([256, 256, 1024])(x)
        x = IdentityBlock([256, 256, 1024])(x)
        x = IdentityBlock([256, 256, 1024])(x)

        # Stage 4
        x = ConvBlock([512, 512, 2048], strides=2)(x)
        x = IdentityBlock([512, 512, 2048])(x)
        x = IdentityBlock([512, 512, 2048])(x)

        # Final layers
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model

    def summary(self):
        """
        Prints a summary of the model architecture.
        """
        self.model.summary()

    def compile_model(self, learning_rate=0.001):
        """
        Compiles the model with specified optimizer, loss function, and metrics.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def call(self, inputs, training=False):
        """
        Defines the forward pass of the model.

        Args:
            inputs: Input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            Output tensor.
        """
        return self.model(inputs, training=training)

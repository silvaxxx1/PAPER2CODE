import tensorflow as tf
from keras.layers import Layer, Conv2D, BatchNormalization, ReLU

class IdentityBlock(Layer):
    """
    The identity block is the standard block used in ResNet when the input and output have the same dimensions.
    """
    def __init__(self, filters):
        """
        Initialize the IdentityBlock.

        Args:
        - filters: A list of three integers, defining the number of filters in the three convolutional layers.
        """
        super(IdentityBlock, self).__init__()

        filters1, filters2, filters3 = filters

        # First convolutional layer
        self.conv1 = Conv2D(filters1, (1, 1), padding='same')
        self.bn1 = BatchNormalization()

        # Second convolutional layer
        self.conv2 = Conv2D(filters2, (3, 3), padding='same')
        self.bn2 = BatchNormalization()

        # Third convolutional layer
        self.conv3 = Conv2D(filters3, (1, 1), padding='same')
        self.bn3 = BatchNormalization()

        self.relu = ReLU()

    def call(self, inputs, training=False):
        """
        Forward pass for the IdentityBlock.

        Args:
        - inputs: Input tensor.
        - training: Boolean, whether the model is in training mode or not.

        Returns:
        - Output tensor.
        """
        # First convolutional layer
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # Add skip connection and apply ReLU
        x += inputs
        x = self.relu(x)

        return x

class ConvBlock(Layer):
    """
    The convolutional block is used in ResNet when the input and output have different dimensions.
    It includes a convolutional layer in the skip connection to match dimensions.
    """
    def __init__(self, filters, strides=2):
        """
        Initialize the ConvBlock.

        Args:
        - filters: A list of three integers, defining the number of filters in the three convolutional layers.
        - strides: An integer, defining the stride to be used in the first convolutional layer and the skip connection.
        """
        super(ConvBlock, self).__init__()

        filters1, filters2, filters3 = filters

        # First convolutional layer
        self.conv1 = Conv2D(filters1, (1, 1), strides=strides, padding='same')
        self.bn1 = BatchNormalization()

        # Second convolutional layer
        self.conv2 = Conv2D(filters2, (3, 3), padding='same')
        self.bn2 = BatchNormalization()

        # Third convolutional layer
        self.conv3 = Conv2D(filters3, (1, 1), padding='same')
        self.bn3 = BatchNormalization()

        self.relu = ReLU()

        # Convolutional layer for the skip connection
        self.conv_res = Conv2D(filters3, (1, 1), strides=strides, padding='same')
        self.bn_res = BatchNormalization()

    def call(self, inputs, training=False):
        """
        Forward pass for the ConvBlock.

        Args:
        - inputs: Input tensor.
        - training: Boolean, whether the model is in training mode or not.

        Returns:
        - Output tensor.
        """
        # First convolutional layer
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # Skip connection with convolution
        res_path = self.conv_res(inputs)
        res_path = self.bn_res(res_path, training=training)

        # Add skip connection and apply ReLU
        x += res_path
        x = self.relu(x)

        return x

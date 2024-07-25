import tensorflow as tf
import json
from data import load_data
from model.ResNet50 import ResNetModel

class Train:
    def __init__(self, config_file):
        """
        Initialize the Train class.

        Args:
            config_file (str): Path to the JSON configuration file.
        """
        self.config = self.load_config(config_file)
        self.train_data, self.test_data = self.load_and_preprocess_data()

    def load_config(self, config_file):
        """
        Load the configuration from the specified JSON file.

        Args:
            config_file (str): Path to the JSON configuration file.

        Returns:
            dict: Loaded configuration.
        """
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config 

    def load_and_preprocess_data(self):
        """
        Load and preprocess the data using the configuration.

        Returns:
            tf.data.Dataset: Preprocessed training dataset.
            tf.data.Dataset: Preprocessed validation dataset.
        """
        train_data, test_data = load_data(image_size=self.config["image_size"], batch_size=self.config["batch_size"])
        return train_data, test_data 

    def train_model(self):
        """
        Train the ResNet model using the loaded and preprocessed data.
        """
        input_shape = (self.config["image_size"], self.config["image_size"], self.config["n_channels"])
        num_classes = 10 

        model = ResNetModel(input_shape, num_classes)
        model.compile_model(learning_rate=self.config["lr"])

        model.fit(self.train_data, epochs=10, validation_data=self.test_data)
        
        loss, accuracy = model.evaluate(self.test_data)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

        model.save("resnet_model.h5")

if __name__ == "__main__":
    trainer = Train("config.json")
    trainer.train_model()

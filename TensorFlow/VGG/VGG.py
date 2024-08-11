#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_datasets as tfds

class Block(tf.keras.Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        
        for i in range(repetitions):
            setattr(self, f'conv2D_{i}', tf.keras.layers.Conv2D(filters=self.filters, 
                                                                kernel_size=self.kernel_size, 
                                                                activation='relu', 
                                                                padding='same'))
        
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)
  
    def call(self, inputs):
        x = getattr(self, 'conv2D_0')(inputs)
        for i in range(1, self.repetitions):
            x = getattr(self, f'conv2D_{i}')(x)
        x = self.max_pool(x)
        return x

class MyVGG(tf.keras.Model):
    def __init__(self, num_classes):
        super(MyVGG, self).__init__()

        # Creating blocks of VGG with the following configurations
        self.block_a = Block(filters=64, kernel_size=3, repetitions=2)
        self.block_b = Block(filters=128, kernel_size=3, repetitions=2)
        self.block_c = Block(filters=256, kernel_size=3, repetitions=3)
        self.block_d = Block(filters=512, kernel_size=3, repetitions=3)
        self.block_e = Block(filters=512, kernel_size=3, repetitions=3)

        # Classification head
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x

def preprocess(features):
    image = tf.image.resize(features['image'], (224, 224))
    return tf.cast(image, tf.float32) / 255., features['label']

def main():
    # Download the dataset
    dataset = tfds.load('cats_vs_dogs', split=tfds.Split.TRAIN, data_dir='data/')

    # Initialize VGG with the number of classes 
    vgg = MyVGG(num_classes=2)

    # Compile with losses and metrics
    vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Apply transformations to dataset
    dataset = dataset.map(preprocess).batch(32)

    # Train the custom VGG model
    vgg.fit(dataset, epochs=10)

if __name__ == "__main__":
    main()

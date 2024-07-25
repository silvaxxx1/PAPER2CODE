# VIT_from_scratch

This repository contains an implementation of Vision Transformer (ViT) from scratch using TensorFlow/Keras. The Vision Transformer is a novel architecture proposed for image classification tasks, which replaces convolutional layers with self-attention mechanisms commonly used in transformers.

## Features

- **Modular Implementation:** The implementation consists of modular components for patch embedding, positional encoding, transformer encoder, multi-layer perceptron (MLP), and assembling components of the Vision Transformer.
- **Configurability:** Configuration parameters such as the number of layers, hidden dimensions, number of heads, patch size, dropout rate, etc., can be easily adjusted to customize the model.
- **Training-ready:** The implementation is designed to be training-ready, allowing users to easily train the Vision Transformer on their own datasets for various image classification tasks.
- **Educational Purpose:** This implementation is ideal for educational purposes, providing insights into the inner workings of Vision Transformers and transformers in general.

## Usage

To use this implementation, simply import the necessary functions/classes from the `vit.py` file into your project. Adjust the configuration parameters as needed and create an instance of the Vision Transformer model using the `vision_transformer` function.


## Acknowledgements

This implementation draws inspiration from various sources, including:

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.
- Implementations and tutorials by deep learning practitioners and educators such as Andrej Karpathy, Andrew Ng, and others.

---

Feel free to customize the README further based on your preferences and additional information you want to provide about your implementation.

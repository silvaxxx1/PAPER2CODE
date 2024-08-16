
# NLP Evolution with PyTorch

This repository provides implementations of various natural language processing (NLP) models using PyTorch, ranging from simple models to advanced architectures. The goal is to explore and understand these models through hands-on coding and experimentation.

## Models Implemented

- **Bigram Model**: A basic model where each character is predicted based on the previous one using a lookup table.
- **MLP (Multi-Layer Perceptron)**: A feedforward neural network with hidden layers.
- **RNN (Recurrent Neural Network)**: A sequential model capturing temporal dependencies.
- **LSTM (Long Short-Term Memory)**: An RNN variant designed to handle long-term dependencies effectively.
- **GRU (Gated Recurrent Unit)**: A simplified version of LSTM with fewer parameters.
- **Transformer**: A model utilizing self-attention mechanisms for capturing long-range dependencies.

## Training and Generation

To train and generate text with these models, you can use the provided `main.py` script. Here's how to run it for different models:

### Example Command

```bash
python main.py --model transformer --data data/input.txt --hidden_size 256 --num_layers 4 --max_new_tokens 100 --block_size 128 --batch_size 64 --epochs 10
```

This command trains the Transformer model using the data from `data/input.txt` and generates text.

## Educational Resource: `from_scratch_for_fun.py`

As an additional resource, we provide a file named `from_scratch_for_fun.py`. This file implements basic versions of MLP, LSTM, and GRU layers using NumPy. While the main focus of this repository is on using PyTorch, this file serves as an educational tool to help you:

- **Understand Core Mechanics**: See how these layers are constructed from scratch.
- **Visualize Computations**: Gain insights into the internal workings of MLP, LSTM, and GRU layers.
- **Experiment**: Modify these basic implementations to explore different aspects of neural networks.

To run the examples in `from_scratch_for_fun.py`, simply execute:

```bash
python from_scratch_for_fun.py
```

## Associated Papers

The current implementations follow the principles outlined in several key papers:

- **Bigram**: "A Simple Statistical Model for Language" (1968)
- **MLP**: Bengio, Y., et al. "Learning Deep Architectures for AI" (2009)
- **RNN**: Mikolov, T., et al. "Recurrent Neural Network based Language Model" (2010)
- **LSTM**: Graves, A. "Supervised Sequence Labelling with Recurrent Neural Networks" (2014)
- **GRU**: Cho, K., et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014)
- **Transformer**: Vaswani, A., et al. "Attention is All You Need" (2017)

Feel free to explore and contribute to this repository. Happy experimenting with NLP models!


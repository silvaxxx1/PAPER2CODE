 
import argparse
import torch
import torch.optim as optim
from models.Bigram import BigramLanguageModel
from models.MLP import MLPLanguageModel  # Assuming MLP class name is MLPLanguageModel
from models.RNN import RNNLanguageModel
from models.LSTM import LSTMLanguageModel  # Assuming LSTM class name is LSTMLanguageModel
from models.GRU import GRULanguageModel  # Assuming GRU class name is GRULanguageModel
from models.Transformer import Transformer  # Adjust import based on your implementation
from data.data import load_data, get_batch, decode

def generate(model, idx, max_new_tokens, stoi, itos):
    """
    Generate text given a model and starting index.

    Args:
        model: The language model to use for generation.
        idx: The initial input tensor.
        max_new_tokens: Number of tokens to generate.
        stoi: String to index mapping.
        itos: Index to string mapping.

    Returns:
        Tensor with generated indices.
    """
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]  # Get logits for the last token
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def train(model_type, data_path, hidden_size, num_layers, max_new_tokens, block_size, batch_size, epochs):
    """
    Train the specified model type on the provided data.

    Args:
        model_type: Type of model to train ('bigram', 'mlp', 'rnn', 'lstm', 'gru', 'transformer').
        data_path: Path to the data file.
        hidden_size: Size of hidden layers.
        num_layers: Number of layers for certain models.
        max_new_tokens: Number of tokens to generate after training.
        block_size: Size of data blocks.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
    """
    # Load and prepare data
    train_data, val_data, stoi, itos = load_data(data_path)
    vocab_size = len(stoi)
    
    # Initialize the model based on the specified type
    if model_type == 'bigram':
        model = BigramLanguageModel(vocab_size)
    elif model_type == 'mlp':
        model = MLPLanguageModel(vocab_size, hidden_size)
    elif model_type == 'rnn':
        model = RNNLanguageModel(vocab_size, hidden_size)
    elif model_type == 'lstm':
        model = LSTMLanguageModel(vocab_size, hidden_size)
    elif model_type == 'gru':
        model = GRULanguageModel(vocab_size, hidden_size)
    elif model_type == 'transformer':
        model = Transformer(n_embd=hidden_size, n_head=8, head_size=64, vocab_size=vocab_size, block_size=block_size, n_layers=num_layers)
    else:
        raise ValueError("Invalid model type")

    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _ in range(len(train_data) // batch_size):
            x_batch, y_batch = get_batch(block_size, batch_size, train_data)
            optimizer.zero_grad()
            logits, loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(train_data) // batch_size)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Generate sample text
        model.eval()
        context = torch.randint(len(train_data) - block_size, (1, block_size))
        generated_idx = generate(model, context, max_new_tokens, stoi, itos)
        print("Generated Text:")
        print(decode(generated_idx[0].tolist(), itos))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Generate with different models")
    parser.add_argument('--model', type=str, choices=['bigram', 'mlp', 'rnn', 'lstm', 'gru', 'transformer'], required=True, help="Model type to train")
    parser.add_argument('--data', type=str, required=True, help="Path to the data file")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden size for MLP, RNN, LSTM, and GRU models")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers for RNN, LSTM, and GRU models")
    parser.add_argument('--max_new_tokens', type=int, default=100, help="Number of tokens to generate")
    parser.add_argument('--block_size', type=int, default=128, help="Block size for data batching")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train(
        model_type=args.model,
        data_path=args.data,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

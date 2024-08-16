import torch

def load_data(path, split=0.9):
    """
    Load and preprocess text data from a file.

    Args:
        path (str): Path to the text file.
        split (float): Ratio for splitting data into training and validation sets.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, dict, dict]: 
            - Training data as tensor
            - Validation data as tensor
            - Dictionary mapping characters to indices
            - Dictionary mapping indices to characters
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(len(data) * split)
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, stoi, itos

def get_batch(block_size, batch_size, data):
    """
    Generate a batch of input and target data.

    Args:
        block_size (int): Length of each sequence.
        batch_size (int): Number of sequences in a batch.
        data (torch.Tensor): Dataset to sample from.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Batch of input data
            - Batch of target data
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def decode(indices, itos):
    """
    Convert a list of indices back into a string.

    Args:
        indices (List[int]): List of indices.
        itos (dict): Dictionary mapping indices to characters.

    Returns:
        str: Decoded string.
    """
    return ''.join([itos[i] for i in indices])

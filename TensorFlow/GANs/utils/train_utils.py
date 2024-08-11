import numpy as np
from PIL import Image


# Utilities

def make_grid(imgs, nrow, padding=0):
    """
    Generate a grid of images.

    Args:
        imgs (numpy.ndarray): The input images with shape (batch, height, width, channels).
        nrow (int): The number of images per row in the grid.
        padding (int, optional): The amount of padding to add around each image in the grid. Defaults to 0.

    Returns:
        numpy.ndarray: The grid of images with shape (height * ncol, width * nrow, channels).

    Raises:
        AssertionError: If the input images do not have a shape of (batch, height, width, channels) 
                        or if nrow is not greater than 0.
    """
    assert imgs.ndim == 4, "Input images must have 4 dimensions: (batch, height, width, channels)"
    assert nrow > 0, "Number of images per row must be greater than 0"

    batch, height, width, ch = imgs.shape
    # Calculate the total number of rows and columns in the grid
    n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow

    # Create padding if needed
    pad = np.zeros((n - batch, height, width, ch), imgs.dtype)
    x = np.concatenate([imgs, pad], axis=0)

    # Add border padding if required
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)),
                   "constant", constant_values=(0, 0))
        height += padding
        width += padding

    # Reshape the images into a grid
    x = x.reshape(ncol, nrow, height, width, ch)
    x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, ch)
    x = x.reshape(height * ncol, width * nrow, ch)

    # Remove extra padding if it was added
    if padding > 0:
        x = x[:(height * ncol - padding), :(width * nrow - padding), :]
        
    return x

def save_img(imgs, filepath, nrow, padding=0):
    """
    Save a grid of images to a file.

    Args:
        imgs (numpy.ndarray): The input images with shape (batch, height, width, channels).
        filepath (str): The path to save the grid image file.
        nrow (int): The number of images per row in the grid.
        padding (int, optional): The amount of padding to add around each image in the grid. Defaults to 0.
    
    Returns:
        None
    """
    grid_img = make_grid(imgs, nrow, padding=padding)
    # Convert the grid image from [-1, 1] to [0, 255] range
    grid_img = ((grid_img + 1.0) * 127.5).astype(np.uint8)
    # Save the image to the specified file path
    with Image.fromarray(grid_img) as img:
        img.save(filepath)

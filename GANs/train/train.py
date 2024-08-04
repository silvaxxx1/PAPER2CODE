import tensorflow as tf
from distrubution import get_strategy
from utils.data import load_data
import tqdm
from GAN.Model import train_step, create_generator, create_discriminator
import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
from utils.train_utils import save_img

# Configuration
resize = 64  # Size to which images will be resized
shape = (resize, resize, 3)  # Shape of the input images (height, width, channels)
z_dim = 128  # Dimensionality of the input noise vector for the generator
n_upsampling = 4  # Number of upsampling layers in the generator
n_downsampling = 4  # Number of downsampling layers in the discriminator
gradient_penalty_mode = 'none'  # Mode for gradient penalty ('none', 'dragan', or 'wgan-gp')

# Set normalization type based on gradient penalty mode
if gradient_penalty_mode == 'none':
    norm = 'batch_norm'  # Use batch normalization if no gradient penalty
elif gradient_penalty_mode in ['dragan', 'wgan-gp']:
    norm = 'layer_norm'  # Use layer normalization if gradient penalty is applied

# Create GAN models
gen_model = create_generator(input_shape=(1, 1, z_dim),
                             output_channels=shape[-1],
                             upsampling_factor=n_upsampling)
dis_model = create_discriminator(input_shape=shape,
                                 n_downsampling=n_downsampling,
                                 norm=norm)

# Setup distribution strategy
strategy = get_strategy()

# Adjust batch size for distributed strategy
batch_size = 8
batch_size = batch_size * strategy.num_replicas_in_sync
dataset = load_data(batch_size)  # Load the dataset using the provided function
dataset = strategy.experimental_distribute_dataset(dataset)  # Distribute dataset for multi-GPU training

# Generate a batch of noise input for evaluation
test_z = tf.random.normal((16, 1, 1, z_dim))

# Training loop
for epoch in range(100):
    with tqdm.tqdm(dataset) as bar:
        bar.set_description(f"Epoch {epoch}")
        for step, (X1, X2) in enumerate(bar):
            d_loss, g_loss, fake = train_step(X1, X2)
            bar.set_postfix({"g_loss": g_loss.numpy(), "d_loss": d_loss.numpy()})
        
        # Generate fake images at the end of each epoch
        fake_images = gen_model(test_z)
    
    # Save generated images
    out_dir = './output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_path = os.path.join(out_dir, f"epoch_{epoch:04}.png")
    save_img(fake_images.numpy()[:64], file_path, 8)

    # Display a gallery of fake faces every epoch
    if epoch % 1 == 0:
        with Image.open(file_path) as img:
            plt.imshow(np.asarray(img))
            plt.show()

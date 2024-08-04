import tensorflow
from train.distrubution import get_strategy , distributed , Reduction
from Generator import create_generator
from Discriminator import create_discriminator
from utils.data import load_data
from utils.model_utils import gradient_penalty 


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

# Gradient penalty weight (used if gradient_penalty_mode is 'dragan' or 'wgan-gp')
gradient_penalty_weight = 10.0

# Build the GAN model
strategy = get_strategy()  # Get the distribution strategy for distributed training
with strategy.scope():  # Scope for distributed training
    # Create the generator model
    gen_model = create_generator(input_shape=(1, 1, z_dim),
                                 output_channels=shape[-1],
                                 upsampling_factor=n_upsampling)
    
    # Create the discriminator model
    dis_model = create_discriminator(input_shape=shape,
                                     n_downsampling=n_downsampling,
                                     norm=norm)
    
    # Optimizers for the generator and discriminator
    gen_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    dis_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    
    # Load and distribute the dataset
    batch_size = 8
    batch_size = batch_size * strategy.num_replicas_in_sync  # Adjust batch size for distributed strategy
    dataset = load_data(batch_size)  # Load the dataset using the provided function
    dataset = strategy.experimental_distribute_dataset(dataset)  # Distribute dataset for multi-GPU training
    
    # Loss function for binary classification (used in GANs)
    loss_func = tensorflow.keras.losses.BinaryCrossentropy(
        from_logits=True,  # Specifies that logits are the input to the loss function
        reduction=tensorflow.keras.losses.Reduction.NONE  # No reduction applied (loss is returned as is)
    )

@distributed(Reduction.SUM, Reduction.SUM, Reduction.CONCAT)
def train_step(real_images1, real_images2):
    """
    Performs a training step for the GAN models.

    Args:
        real_images1 (tf.Tensor): The first batch of real images.
        real_images2 (tf.Tensor): The second batch of real images.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: The loss values for the discriminator and generator, 
                                                 and the generated fake images.

    Steps:
        1. Concatenates the real image batches into a single tensor.
        2. Generates fake images using the generator model.
        3. Computes the discriminator loss based on real and fake images.
        4. Calculates the gradient penalty (if applicable) and adds it to the discriminator loss.
        5. Updates the discriminator model weights.
        6. Computes the generator loss based on the fake images.
        7. Updates the generator model weights.
    """
    # Concatenate real images
    all_images = tensorflow.concat([real_images1, real_images2], axis=0)
    
    # Phase 1 - Train the discriminator
    with tensorflow.GradientTape() as d_tape:
        z = tensorflow.random.normal((all_images.shape[0], 1, 1, z_dim))
        fake_images = gen_model(z)
        # Get discriminator predictions
        fake_logits = dis_model(fake_images)
        real_logits = dis_model(all_images)
        # Calculate discriminator loss
        d_loss = 0.5 * (loss_func(tensorflow.ones_like(real_logits), real_logits) +
                        loss_func(tensorflow.zeros_like(fake_logits), fake_logits))
        
        # Gradient Penalty (if applicable)
        gp = gradient_penalty(lambda: dis_model(dis_model, training=True), all_images, fake_images, mode=gradient_penalty_mode)
        gp = gp / (batch_size * 2)
        d_loss = d_loss + gp * gradient_penalty_weight
        
    # Get gradients and update discriminator weights
    gradients = d_tape.gradient(d_loss, dis_model.trainable_variables)
    dis_optimizer.apply_gradients(zip(gradients, dis_model.trainable_variables))
    
    # Phase 2 - Train the generator
    with tensorflow.GradientTape() as g_tape:
        z = tensorflow.random.normal((all_images.shape[0], 1, 1, z_dim))
        fake_images = gen_model(z)
        # Get generator loss
        fake_logits = dis_model(fake_images)
        g_loss = loss_func(tensorflow.ones_like(fake_logits), fake_logits)
    
    # Get gradients and update generator weights
    gradients = g_tape.gradient(g_loss, gen_model.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients, gen_model.trainable_variables))
    
    return d_loss, g_loss, fake_images

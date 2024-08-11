import tensorflow as tf
from loss import *
from data import *
from IPython.display import display as display_fn
from IPython.display import Image, clear_output
from inception import download_inception_weights


def update_gradient(image,style_target,style_weight,
    content_target,content_weight,optimizer):
    
    with tf.GradientTape() as tape:
       
       style_features = style_feats(image)
       content_features = content_feats(image)
       
       loss = total_loss(style_target,style_features,content_target,content_features,style_weight,content_weight)
       
       grads = tape.gradient(loss,image)
       
       optimizer.apply_gradients([(grads,image)])
       
       image.assign(clip_image(image, min_value=0.0, max_value=255.0))   
      

def fit_style_transfer_train(style_image, content_image, style_weight=1e-2, content_weight=1e-4,
                       optimizer=None, epochs=1, steps_per_epoch=1):   
    
    images  = []
    step = 0 
    
    style_targets = style_feats(style_image) 
    content_targets = content_feats(content_image)
    
    gen_image = tf.Variable(tf.cast(content_image, tf.float32))

    images.append(tf.convert_to_tensor(content_image))
    
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            update_gradient(gen_image,style_targets,style_weight,content_targets,content_weight,optimizer)
            
            print(".", end='')
            if (m + 1) % 10 == 0:
                images.append(tf.convert_to_tensor(generated_image.numpy()))  # Store the generated image

        # Display the current stylized image
        clear_output(wait=True)
        display_image = tensor_to_image(generated_image)
        display_fn(display_image)

        # Append to the image collection for visualization later
        images.append(tf.convert_to_tensor(generated_image.numpy()))
        print("Train step: {}".format(step))

    # Convert to uint8 (expected dtype for images with pixels in the range [0,255])
    generated_image = tf.cast(generated_image, dtype=tf.uint8)

    return generated_image, images



import tensorflow as tf
from data import load_style_and_content
from data import show_images_with_objects 
from  train import fit_style_transfer_train
import matplotlib.pyplot as plt
from inception import incprtion_model , choose_layers


style_path = r"C:\Users\USER\SILVA\PAPER2CODE\TensorFlow\NeuralStyleTransfer\me.jpg"
con_path = r"C:\Users\USER\SILVA\PAPER2CODE\TensorFlow\NeuralStyleTransfer\style.jpeg"
style_image , content_image  = load_style_and_content(style_path , con_path)


show_images_with_objects(images=[style_image , content_image],
                         titles=["Style Image" , "Content Image"])


# PLEASE DO NOT CHANGE THE SETTINGS HERE

# define style and content weight
style_weight =  1
content_weight = 1e-32 

# define optimizer. learning rate decreases per epoch.
adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=80.0, decay_steps=100, decay_rate=0.80
    )
)

# start the neural style transfer
stylized_image, display_images = fit_style_transfer_train(style_image=style_image, content_image=content_image, 
                                                    style_weight=style_weight, content_weight=content_weight,
                                                    optimizer=adam, epochs=10, steps_per_epoch=100)
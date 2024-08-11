import tensorflow as tf

def choose_layers():
    content_layers = ["conv2d_88"]  # Use actual layer names
    style_layers = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"]  # Use actual layer names

    all_layers  = content_layers + style_layers

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    
    return num_content_layers, num_style_layers, all_layers

def incprtion_model(all_layers):
    inception  = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    inception.trainable = False
    
    # Correctly access layers by name
    output_layers = [inception.get_layer(name).output for name in all_layers]
    
    model = tf.keras.Model(inputs=inception.input, outputs=output_layers)
    
    return model 

def download_inception_weights():
    _, _, all_layers = choose_layers()
    inception_model = incprtion_model(all_layers)
    return inception_model 

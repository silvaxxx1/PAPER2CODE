import tensorflow as tf 
from data import perpocess_image
from inception import incprtion_model 
from inception import choose_layers 


num_content_layers, num_style_layers,_= choose_layers()


def style_loss(feats, targets):
    style_loss = tf.reduce_mean(tf.square(feats - targets))
    return style_loss

def content_loss(feats, targets):
    content_loss = 0.5 * tf.reduce_sum(tf.square(feats - targets))
    return content_loss

def gram_matrix(x):
    gram = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    
    x_shape = tf.shape(x)
    height  = x_shape[1]
    width   = x_shape[2]
    
    return gram / tf.cast(height * width, tf.float32) 
    
def style_feats(image):
    style_image_processed =perpocess_image(image)
    out =  incprtion_model(style_image_processed)
    style_outs = out[num_content_layers:]
    gram_style_feature = [gram_matrix(style) for style in style_outs]
    
    return gram_style_feature

def content_feats(image):
    content_image_processed =perpocess_image(image)
    out =  incprtion_model(content_image_processed)
    content_outs = out[:num_content_layers]
    
    return content_outs


def total_loss(style_targets,style_outs,content_targets,content_outs,style_weight,content_weight,content_loss_weight,style_loss_weight):
    
    style_loss  = tf.add_n([
        style_loss(style_outs,style_targets)
        for style_targets,style_outs in zip(style_targets,style_outs)
    ])
    
    content_loss = tf.add_n([
        content_loss(content_outs,content_targets)
        for content_targets,content_outs in zip(content_targets,content_outs)
    ])
    
    style_loss *= style_loss_weight
    content_loss *= content_loss_weight
    total_loss = style_loss + content_loss
    
    return total_loss


    
    
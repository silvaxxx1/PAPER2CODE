import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

class_names = ['sky', 'building', 'column/pole', 'road', 'sidewalk', 'vegetation', 'traffic light',
               'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void']

colors = sns.color_palette(None, len(class_names))

def PIL_image(images):
    '''Combines multiple images into a single PIL image, arranged horizontally.'''
    h = [image.shape[0] for image in images]
    w = [image.shape[1] for image in images]
    h_max = max(h)
    w_sum = sum(w)
    
    new_image = Image.new('RGB', (w_sum, h_max))
    offset = 0
    for image in images:
        PIL_im = Image.fromarray(np.uint8(image))
        new_image.paste(PIL_im, (offset, 0))
        offset += image.shape[1]
        
    return new_image 

def annotations__colors(annotations):
    '''Converts label map annotations into a colorized segmentation image.'''
    h, w = annotations.shape
    seg_image = np.zeros((h, w, 3)).astype('float')
    
    for c in range(len(class_names)):
        seg = (annotations == c)
        seg_image[:, :, 0] += seg * (colors[c][0] * 255.0)
        seg_image[:, :, 1] += seg * (colors[c][1] * 255.0)
        seg_image[:, :, 2] += seg * (colors[c][2] * 255.0)
        
    return seg_image.astype('uint8')

def show_predictions(image, labelmaps, titles, iou_list, dice_score_list, save_path=None):
    '''Displays the input image, true annotation, and predicted annotation side by side.'''
    true_img = annotations__colors(labelmaps[1])
    pred_img = annotations__colors(labelmaps[0])
    
    image = (image + 1) * 127.5
    images = [image.astype(np.uint8), true_img, pred_img]
    
    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list))]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)
    
    display_string_list = ["{}: IOU: {:.2f}, DICE_SCORE: {:.2f}".format(class_names[idx], iou, dice_score)
                           for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list)
    
    plt.figure(figsize=(15, 4))
    
    for idx, image in enumerate(images):
        plt.subplot(1, 3, idx + 1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(image)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def show_annotation_and_image(image, annotation, save_path=None):
    '''Displays the input image and its corresponding annotation side by side.'''
    new_annotation = np.argmax(annotation, axis=2)
    seg_image = annotations__colors(new_annotation)
    
    image = (image + 1) * 127.5
    image = image.astype(np.uint8)
    images = [image, seg_image]
    
    fused_image = PIL_image(images)
    if save_path:
        fused_image.save(save_path)
    plt.imshow(fused_image)

def show_annotation_and_image_list(dataset, save_dir=None):
    '''Displays a list of images and their corresponding annotations.'''
    ds = dataset.unbatch()
    ds = ds.shuffle(buffer_size=100)
    
    plt.figure(figsize=(25, 15))
    plt.title("Image and Annotation")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)
    
    for idx, (image, annotation) in enumerate(ds.take(9)):
        plt.subplot(3, 3, idx + 1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy())
    
    if save_dir:
        plt.savefig(save_dir)
    plt.show()

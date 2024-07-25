import numpy as np
from data import get_dataset_slice_paths, get_validation_dataset
from keras.models import load_model
from vis import show_predictions

class_names = [
    'sky', 'building', 'column/pole', 'road', 'sidewalk', 'vegetation',
    'traffic light', 'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void'
]

validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(
    r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\annotations_prepped_test',
    r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\images_prepped_test'
)

validation_datasets = get_validation_dataset(validation_image_paths, validation_label_map_paths)

def get_images_and_seg_test(test_counts=64):
    y_true_seg = []
    y_true_img = []
    
    ds = validation_datasets.unbatch()
    ds = ds.batch(101)
    
    for image, annotations in ds.take(1):
        y_true_img = image
        y_true_seg = annotations 
        
        y_true_seg = y_true_seg[:test_counts, :, :, :]
        y_true_seg = np.argmax(y_true_seg, axis=3) 
        
        return y_true_img, y_true_seg

def computer_metrics(y_true, y_pred):
    iou_list = []
    dice_score_list = []
    smoothening_factor = 0.00001
    
    for i in range(len(class_names)):
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        overlap = np.sum((y_true == i) * (y_pred == i))
        union = y_true_area + y_pred_area - overlap

        iou = (overlap + smoothening_factor) / (union + smoothening_factor)
        iou_list.append(iou)
        dice_score = 2 * ((overlap + smoothening_factor) / (y_true_area + y_pred_area + smoothening_factor))
        dice_score_list.append(dice_score)
    
    return iou_list, dice_score_list

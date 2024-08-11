import tensorflow as tf 
import numpy as np 
from Model import unet
from data import test_dataset , get_dataset
from prediction import get_test_image_and_annotation_arrays , make_predictions 
from utils import class_wise_metrics , display_with_metrics 


_ , info = get_dataset()
model = unet()
 
 
class_names = ['pet', 'background', 'outline']
BATCH_SIZE = 64
BUFFER_SIZE = 1000
# Setup the ground truth and predictions.

# get the ground truth from the test set
y_true_images, y_true_segments = get_test_image_and_annotation_arrays()

# feed the test set to th emodel to get the predicted masks
results = model.predict(test_dataset, 
                        steps=info.splits['test'].num_examples // BATCH_SIZE)
results = np.argmax(results, axis=3)
results = results[..., tf.newaxis]

cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y_true_segments, results)


# show the IOU for each class
for idx, iou in enumerate(cls_wise_iou):
    spaces = ' ' * (10-len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, iou)) 
    
    
# show the Dice Score for each class
for idx, dice_score in enumerate(cls_wise_dice_score):
    spaces = ' ' * (10-len(class_names[idx]) + 2)
    print("{}{}{} ".format(class_names[idx], spaces, dice_score)) 
    
# Please input a number between 0 to 3647 to pick an image from the dataset
integer_slider = 3646

# Get the prediction mask
y_pred_mask = make_predictions(y_true_images[integer_slider], y_true_segments[integer_slider])

# Compute the class wise metrics
iou, dice_score = class_wise_metrics(y_true_segments[integer_slider], y_pred_mask)  

# Overlay the metrics with the images
display_with_metrics([y_true_images[integer_slider], y_pred_mask, y_true_segments[integer_slider]], iou, dice_score)
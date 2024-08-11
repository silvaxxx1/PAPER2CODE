from UTILS.metrics import computer_metrics, get_images_and_seg_test
from UTILS.vis import show_predictions
from keras.models import load_model
import numpy as np
from data.data import get_dataset_slice_paths, get_validation_dataset

# Define paths to the validation data
validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(
    'data/annotations_prepped_test',
    'data/images_prepped_test'
)

# Load validation dataset
validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

# Define class names
class_names = [
    'sky', 'building', 'column/pole', 'road', 'sidewalk', 'vegetation',
    'traffic light', 'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void'
]

# Get ground truth images and segmentations
y_true_img, y_true_seg = get_images_and_seg_test()

# Load the trained model
model = load_model("results/checkpoints/final_model.h5")

# Predict on validation dataset
results = model.predict(validation_dataset)
results = np.argmax(results, axis=3)

# Pick an image index from the validation set
integer_slider = 0

# Compute metrics for the selected image
iou, dice_score = computer_metrics(y_true_seg[integer_slider], results[integer_slider])

# Visualize the output and metrics
show_predictions(
    y_true_img[integer_slider],
    [results[integer_slider], y_true_seg[integer_slider]],
    ["Image", "Predicted Mask", "True Mask"],
    iou, dice_score
)

# Compute class-wise metrics
cls_wise_iou, cls_wise_dice_score = computer_metrics(y_true_seg, results)

# Print IOU and Dice score for each class
print("Class-wise IOU:")
for idx, iou in enumerate(cls_wise_iou):
    spaces = ' ' * (13 - len(class_names[idx]) + 2)
    print(f"{class_names[idx]}{spaces}{iou:.4f}")

print("\nClass-wise Dice Score:")
for idx, dice_score in enumerate(cls_wise_dice_score):
    spaces = ' ' * (13 - len(class_names[idx]) + 2)
    print(f"{class_names[idx]}{spaces}{dice_score:.4f}")

import tensorflow as tf 
from Model import unet
from data import  training_dataset, test_dataset , get_dataset 
from utils import plot_metrics 

dataset , info = get_dataset() 
train_dataset =  training_dataset()
test_dataset = test_dataset()
model = unet()

BATCH_SIZE = 64
BUFFER_SIZE = 1000

# configure the optimizer, loss and metrics for training
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# configure the training parameters and train the model

TRAIN_LENGTH = info.splits['train'].num_examples
EPOCHS = 10
VAL_SUBSPLITS = 5
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

model_history = model.fit(train_dataset, 
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset)

plot_metrics("loss", title="Training vs Validation Loss", ylim=1)

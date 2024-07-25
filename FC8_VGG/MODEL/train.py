import tensorflow as tf
from data import get_dataset_slice_paths, get_training_dataset, get_validation_dataset
from Model import seg_model
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_metrics(history, save_dir):
    """Plot training & validation accuracy and loss."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'training_plots.png'))
    plt.show()

def calculate_metrics(dataset, model, batch_size, steps_per_epoch):
    """Calculate loss and accuracy for a given dataset."""
    loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.Mean()
    
    for step, (images, labels) in enumerate(dataset.batch(batch_size)):
        logits = model(images, training=False)
        loss = tf.keras.losses.categorical_crossentropy(labels, logits)
        loss_metric.update_state(tf.reduce_mean(loss))
        
        accuracy = tf.keras.metrics.categorical_accuracy(labels, logits)
        accuracy_metric.update_state(tf.reduce_mean(accuracy))
        
        if step >= steps_per_epoch - 1:
            break
    
    return loss_metric.result().numpy(), accuracy_metric.result().numpy()

def main(train_image_path, train_label_path, val_image_path, val_label_path, batch_size=64, epochs=170, learning_rate=1E-2, momentum=0.9, save_dir='.'):
    # Get the paths to the images
    training_image_paths, training_label_map_paths = get_dataset_slice_paths(train_label_path, train_image_path)
    validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(val_label_path, val_image_path)

    training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
    validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

    BATCH_SIZE = batch_size
    EPOCHS = epochs
    train_count = len(training_image_paths)
    validation_count = len(validation_image_paths)

    steps_per_epoch = train_count // BATCH_SIZE
    validation_steps = validation_count // BATCH_SIZE

    model = seg_model()
    sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        # Training
        epoch_loss = 0
        epoch_accuracy = 0
        for step, (images, labels) in enumerate(training_dataset.batch(BATCH_SIZE)):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = tf.keras.losses.categorical_crossentropy(labels, logits)
                loss = tf.reduce_mean(loss)
            
            grads = tape.gradient(loss, model.trainable_variables)
            sgd.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            accuracy = tf.keras.metrics.categorical_accuracy(labels, logits)
            epoch_accuracy += tf.reduce_mean(accuracy).numpy()
        
        epoch_loss /= steps_per_epoch
        epoch_accuracy /= steps_per_epoch
        
        # Validation
        val_loss, val_accuracy = calculate_metrics(validation_dataset, model, BATCH_SIZE, validation_steps)
        
        # Save history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save model checkpoint
        model.save_weights(os.path.join(save_dir, f'model_checkpoint_epoch_{epoch + 1}.h5'))
    
    # Save final model
    model.save(os.path.join(save_dir, 'final_model.keras'))
    
    # Save training history
    with open(os.path.join(save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Plot and save metrics
    plot_metrics(history, save_dir)

if __name__ == '__main__':
    train_image_path = r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\images_prepped_train'
    train_label_path = r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\annotations_prepped_train'
    val_image_path = r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\images_prepped_test'
    val_label_path = r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\dataset\dataset1\annotations_prepped_test'
    batch_size = 64
    epochs = 170
    learning_rate = 1E-2
    momentum = 0.9
    save_dir = r'C:\Users\USER\SILVA\PAPER2CODE\FC8_VGG\results'

    os.makedirs(save_dir, exist_ok=True)

    main(train_image_path, train_label_path, val_image_path, val_label_path, batch_size, epochs, learning_rate, momentum, save_dir)

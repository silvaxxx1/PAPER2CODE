import tensorflow as tf 
from vit import vision_transformer
import numpy as np 

config = {
        "num_layers": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "num_patches": 256,
        "patch_size": 32,
        "num_channels": 3
    }

# Create dummy data
num_samples = 100
num_classes = 10
input_shape = (config["num_patches"], config["patch_size"] * config["patch_size"] * config["num_channels"])
X_dummy = np.random.rand(num_samples, *input_shape).astype(np.float32)
y_dummy = np.random.randint(0, num_classes, size=(num_samples,))
y_dummy = tf.keras.utils.to_categorical(y_dummy, num_classes=num_classes)
dummy_dataset = tf.data.Dataset.from_tensor_slices((X_dummy, y_dummy)).batch(32)

# Create and compile the model
model = vision_transformer(config)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with dummy data
history = model.fit(dummy_dataset, epochs=1, steps_per_epoch=5)

# Evaluate the model with dummy data
test_loss, test_acc = model.evaluate(dummy_dataset)
print(f'Test Accuracy: {test_acc:.4f}')

# Make predictions with dummy data
predictions = model.predict(dummy_dataset)
print(predictions)
# =========================================
# REGULARIZED CNN – CATS vs DOGS (TFDS)
# =========================================

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------
# 1. Load Inbuilt Dataset (TFDS)
# -------------------------------
(dataset_train, dataset_val), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

# -------------------------------
# 2. Image Preprocessing
# -------------------------------
IMG_SIZE = 150

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

dataset_train = (
    dataset_train
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000)
    .batch(32)
    .prefetch(AUTOTUNE)
)

dataset_val = (
    dataset_val
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(32)
    .prefetch(AUTOTUNE)
)

# -------------------------------
# 3. Visualize Sample Images
# -------------------------------
plt.figure(figsize=(8,8))
for images, labels in dataset_train.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title("Dog" if labels[i] else "Cat")
        plt.axis('off')
plt.suptitle("Sample Images from TFDS")
plt.show()

# -------------------------------
# 4. REGULARIZED CNN MODEL
# -------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()

# -------------------------------
# 5. Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 6. Train Model
# -------------------------------
history = model.fit(
    dataset_train,
    validation_data=dataset_val,
    epochs=5,
    steps_per_epoch=200,        
    validation_steps=50,
    verbose=2
)

# -------------------------------
# 7. Accuracy & Loss Plots
# -------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# -------------------------------
# 8. TESTING WITH A SAMPLE IMAGE
# -------------------------------
for sample_image, sample_label in dataset_val.take(1):
    test_img = sample_image[0]
    true_label = sample_label[0]

test_img_exp = tf.expand_dims(test_img, axis=0)
prediction = model.predict(test_img_exp)[0][0]

plt.imshow(test_img)
plt.title(f"Predicted: {'Dog' if prediction > 0.5 else 'Cat'} | Probability: {prediction:.2f}")
plt.axis('off')
plt.show()

# -------------------------------
# 9. Architecture Diagram
# -------------------------------
from tensorflow.keras.utils import plot_model
plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
    expand_nested=False
)
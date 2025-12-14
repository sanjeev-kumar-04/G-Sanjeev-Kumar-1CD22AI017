# ===============================
# 1. Imports
# ===============================
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# 2. Load MALARIA dataset from TFDS
# ===============================
(ds_train, ds_val), ds_info = tfds.load(
    "malaria",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

print(ds_info)

# ===============================
# 3. Dataset parameters
# ===============================
IMG_SIZE = 150
BATCH_SIZE = 20

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_ds = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE)
val_ds   = ds_val.map(preprocess).batch(BATCH_SIZE)

# ===============================
# 4. Visualize sample images (like cats vs dogs)
# ===============================
plt.figure(figsize=(8,8))
for images, labels in train_ds.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.title("Parasitized" if labels[i] == 1 else "Uninfected")
        plt.axis("off")
plt.show()

# ===============================
# 5. CNN Model (same structure idea)
# ===============================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# ===============================
# 6. Compile
# ===============================
model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 7. Train
# ===============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    verbose=2
)

# ===============================
# 8. Predict a random image
# ===============================
for images, labels in val_ds.take(1):
    idx = random.randint(0, images.shape[0]-1)
    img = images[idx]
    label = labels[idx]

img_input = np.expand_dims(img, axis=0)
prediction = model.predict(img_input)

plt.imshow(img)
plt.title("Predicted: Parasitized" if prediction[0] > 0.5 else "Predicted: Uninfected")
plt.axis("off")
plt.show()

print("Raw prediction value:", prediction[0][0])

# ===============================
# 9. Feature Map Visualization (same logic)
# ===============================
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.inputs, outputs=successive_outputs)

x = img_input
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]

        display_grid = np.zeros((size, size * n_features))

        for i in range(n_features):
            fm = feature_map[0, :, :, i]
            fm -= fm.mean()
            if fm.std() != 0:
                fm /= fm.std()
            fm *= 64
            fm += 128
            fm = np.clip(fm, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = fm

        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

# ===============================
# 10. Accuracy & Loss Plots
# ===============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.show()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()

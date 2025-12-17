# -*- coding: utf-8 -*-
"""
Character-Level Text Generation using LSTM (Improved for stability and faster execution)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.utils import shuffle

# -------------------------------
# 1. New Example Text (Different Use Case)
# -------------------------------
text = "machine learning models learn patterns from data and improve over time"

# -------------------------------
# 2. Character Encoding
# -------------------------------
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# -------------------------------
# 3. Sequence Creation (Updated to Use Larger seq_length)
# -------------------------------
seq_length = 10  # Increased sequence length for capturing longer dependencies
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[c] for c in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

# One-hot encoding for sequences and labels
X_one_hot = tf.one_hot(X, len(chars))
y_one_hot = tf.one_hot(y, len(chars))

# -------------------------------
# 4. Model Definition (LSTM Layer instead of SimpleRNN)
# -------------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(seq_length, len(chars)), activation='tanh'))  # Changed to LSTM with 64 units
model.add(Dense(len(chars), activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Lowered learning rate for better stability
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 5. Faster Training (More Epochs and Shuffling Data)
# -------------------------------
# Shuffling training data to avoid overfitting on any particular sequence
X, y = shuffle(X, y, random_state=42)

# Training for more epochs (200 epochs)
model.fit(
    X_one_hot,
    y_one_hot,
    epochs=200,  # Increased number of epochs
    batch_size=16,
    verbose=2
)

# -------------------------------
# 6. Text Generation (Corrected & Stable with Longer Sequences)
# -------------------------------
start_seq = "machine learing "  # Starting sequence
generated_text = start_seq

for _ in range(100):
    input_seq = generated_text[-seq_length:]

    # Padding if sequence is shorter than seq_length
    if len(input_seq) < seq_length:
        input_seq = " " * (seq_length - len(input_seq)) + input_seq

    input_idx = [char_to_index.get(c, 0) for c in input_seq]
    input_idx = np.array(input_idx).reshape(1, seq_length)
    input_one_hot = tf.one_hot(input_idx, len(chars))

    prediction = model.predict(input_one_hot, verbose=0)
    next_char = index_to_char[np.argmax(prediction)]
    generated_text += next_char

print("\nGenerated Text:\n")
print(generated_text)

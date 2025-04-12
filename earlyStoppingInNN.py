# keras_early_stopping.py

# ✅ Example of using Early Stopping in a Neural Network with Keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# Step 1: Create a simple binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Step 2: Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Define early stopping
early_stop = EarlyStopping(
    monitor='val_loss',        # Watch validation loss
    patience=3,                # Wait for 3 epochs before stopping if no improvement
    restore_best_weights=True # Load the best weights when stopping
)

# Step 6: Train the model with early stopping
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,               # Max 100 epochs
    callbacks=[early_stop],   # Use early stopping
    verbose=1
)

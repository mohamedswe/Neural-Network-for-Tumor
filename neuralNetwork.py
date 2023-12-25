
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv("/content/heart_failure_clinical_records_dataset.csv")

# Split the data into features (X) and target (y)
X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#normalizing the data to improve accuracy starting accuracy before implementation is 70% after implementation 87%
scaler = StandardScaler()
X_train_normal = scaler.fit_transform(X_train)
X_test_normal =  scaler.fit_transform(X_test)

# Create a Sequential model
model = tf.keras.models.Sequential()

# Add layers to the model current layers 2 initial accracy 70%
model.add(tf.keras.layers.Dense(32, input_shape=(X_train_normal.shape[1],), activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # Use sigmoid activation for binary classification

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss')

model.fit(X_train_normal, y_train, epochs=200, validation_split=0.1, callbacks = [early_stopping, model_checkpoint])

model.evaluate(X_test_normal, y_test)
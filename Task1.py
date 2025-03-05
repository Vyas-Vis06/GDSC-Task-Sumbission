# Step 1: Setup & Import Libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Enable Mixed Precision for Faster Training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Enable TPU Strategy if available
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(resolver)
    print("Using TPU for training")
except ValueError:
    strategy = tf.distribute.MirroredStrategy()
    print("Using GPU(s) for training")

# Step 2: Load & Preprocess Data

# Load the CIFAR-10 dataset from Keras datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the pixel values to be between 0 and 1 for better convergence
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Step 3: Data Augmentation

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

# Step 4A: Implement Transfer Learning with VGG16
with strategy.scope():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Freeze base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', dtype='float32')(x)  # Prevent NaN issues

    vgg_model = Model(inputs=base_model.input, outputs=output)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    vgg_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Step 4B: Implement Custom CNN Model
with strategy.scope():
    cnn_model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])

    cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train & Validate Models

# Implement early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the VGG16 model using augmented data
vgg_history = vgg_model.fit(datagen.flow(x_train, y_train, batch_size=64),
                            epochs=20, validation_data=(x_test, y_test),
                            callbacks=[early_stopping, lr_scheduler])

# Train the CNN model
cnn_history = cnn_model.fit(datagen.flow(x_train, y_train, batch_size=64),
                            epochs=20, validation_data=(x_test, y_test),
                            callbacks=[early_stopping, lr_scheduler])

# Step 6: Evaluate Performance

# Plot training and validation accuracy/loss curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(vgg_history.history['accuracy'], label='VGG16 Train Accuracy')
plt.plot(vgg_history.history['val_accuracy'], label='VGG16 Val Accuracy')
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.subplot(1,2,2)
plt.plot(vgg_history.history['loss'], label='VGG16 Train Loss')
plt.plot(vgg_history.history['val_loss'], label='VGG16 Val Loss')
plt.plot(cnn_history.history['loss'], label='CNN Train Loss')
plt.plot(cnn_history.history['val_loss'], label='CNN Val Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

# Evaluate VGG16 Model
y_pred_vgg = np.argmax(vgg_model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("VGG16 Classification Report:")
print(classification_report(y_true, y_pred_vgg))

precision_vgg, recall_vgg, f1_vgg, _ = precision_recall_fscore_support(y_true, y_pred_vgg, average='weighted')
print(f"VGG16 Precision: {precision_vgg:.4f}")
print(f"VGG16 Recall: {recall_vgg:.4f}")
print(f"VGG16 F1-score: {f1_vgg:.4f}")

# Evaluate CNN Model
y_pred_cnn = np.argmax(cnn_model.predict(x_test), axis=1)
print("CNN Classification Report:")
print(classification_report(y_true, y_pred_cnn))

precision_cnn, recall_cnn, f1_cnn, _ = precision_recall_fscore_support(y_true, y_pred_cnn, average='weighted')
print(f"CNN Precision: {precision_cnn:.4f}")
print(f"CNN Recall: {recall_cnn:.4f}")
print(f"CNN F1-score: {f1_cnn:.4f}")

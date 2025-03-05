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

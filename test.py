import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import quantus
from quantus import FaithfulnessCorrelation, RelativeInputStability, Sparsity

print("All libraries imported successfully!")

# import tensorflow as tf

# Check if TensorFlow is using Metal
print("Built with CUDA:", tf.test.is_built_with_cuda())  # Should return False for tf-macos
print("Available GPUs:", tf.config.list_physical_devices('GPU'))  # Should list your Apple GPU

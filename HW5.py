# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from quantus import FaithfulnessCorrelation, RelativeInputStability # Sparsity
import quantus

# Load Fashion MNIST dataset
def load_fmnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize and reshape the data for the CNN
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test

# Define the CNN model
def create_lenet_model():
    model = Sequential([
        Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=2),
        Conv2D(64, kernel_size=3, activation="relu"),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Train and evaluate the model
def train_and_evaluate_model():
    x_train, y_train, x_test, y_test = load_fmnist()
    model = create_lenet_model()
    model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}")
    return model, x_test, y_test

# Get correctly classified samples
def get_correctly_classified_samples(model, x_test, y_test):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    correct_indices = np.where(predicted_labels == y_test)[0]
    return x_test[correct_indices], y_test[correct_indices]

# Apply explainability methods and generate heatmaps
def apply_explainability_methods(model, x_correct, y_correct, methods):
    explanations = {}
    num_samples = len(x_correct)

    # Generate explanations for each method
    for method_name, method in methods.items():
        print(f"Generating explanations for {method_name}...")
        explanations[method_name] = [method(model=model, inputs=x, targets=y) for x, y in zip(x_correct[:1000], y_correct[:1000])]

    # for method in method:
    #     print(f"Generating explanations for {method_name}...")
    #     explanations[method_name] = [method(model=model, inputs=x, targets=y) for x, y in zip(x_correct[:1000], y_correct[:1000])]


    return explanations

# Visualize heatmaps
def plot_heatmaps(x_samples, explanations, method_names):
    plt.figure(figsize=(10, 5))
    for i in range(len(x_samples)):
        plt.subplot(2, len(x_samples), i + 1)
        plt.imshow(x_samples[i].reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.axis("off")
        for j, method_name in enumerate(method_names):
            plt.subplot(2, len(x_samples), len(x_samples) * (j + 1) + i + 1)
            plt.imshow(explanations[method_name][i], cmap="hot")
            plt.title(method_name)
            plt.axis("off")
    plt.show()

# Quantitative analysis
def perform_quantitative_analysis(model, x_correct, y_correct, methods):
    results = {}
    quantus_metrics = {
        "Faithfulness": FaithfulnessCorrelation,
        "Stability": RelativeInputStability,
        "Sparsity": Sparsity
    }

    # Calculate metrics for each method
    for method_name, method in methods.items():
        print(f"Calculating metrics for {method_name}...")
        results[method_name] = {}
        for metric_name, metric_cls in quantus_metrics.items():
            metric = metric_cls(model=model, xai_method=method_name)
            results[method_name][metric_name] = metric(x_correct[:1000], y_correct[:1000])

    return results

# Main function
def main():
    # Step 1: Train and evaluate the model
    model, x_test, y_test = train_and_evaluate_model()

    # Step 2: Get correctly classified samples
    x_correct, y_correct = get_correctly_classified_samples(model, x_test, y_test)

    # Step 3: Define explainability methods
    method_list = quantus.AVAILABLE_XAI_METHODS_CAPTUM
    methods = {
        "Gradient": quantus.Gradient,
        "LIME": quantus.Lime,
        "InputXGradient": quantus.InputXGradient,
        "IntegratedGradients": quantus.IntegratedGradients,
    }

    # Step 4: Apply explainability methods and generate explanations
    explanations = apply_explainability_methods(model, x_correct, y_correct, methods)

    # Step 5: Visualize heatmaps
    plot_heatmaps(x_correct[:5], explanations, list(methods.keys()))

    # Step 6: Perform quantitative analysis
    results = perform_quantitative_analysis(model, x_correct, y_correct, methods)
    print("Quantitative Analysis Results:")
    print(results)

# Run the main function
if __name__ == "__main__":
    main()

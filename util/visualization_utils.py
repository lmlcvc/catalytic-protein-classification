import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tabulate import tabulate

import tensorflow as tf


@tf.function
def get_gradients(model, inputs):
    # Function to get the gradients of the output predictions with respect to the input graph nodes
    x_t, mask, A_m = inputs  # Unpack the input tensors

    mask_float = tf.where(mask, tf.ones_like(mask, dtype=tf.float32),
                          tf.zeros_like(mask, dtype=tf.float32))

    inputs_adapted = [x_t, mask_float, A_m]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs_adapted)
        predictions = model(inputs)

    gradients = (tape.gradient(predictions, x_t), tape.gradient(predictions, A_m))

    return gradients


def calculate_saliency(gradients):
    saliency_map = np.abs(gradients[0].numpy())
    saliency_map /= np.max(saliency_map)
    saliency_map = np.transpose(saliency_map)

    return saliency_map


def visualize_heatmap(heatmap, filename, figsize=(8, 8), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', aspect='auto')
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Saliency')
    plt.xlabel('Node Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def evaluate_model(predictions, labels):
    predictions = [prediction for sublist in predictions for prediction in sublist]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)

    metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
    metric_values = [accuracy, precision, recall, f1, roc_auc]

    metric_rows = [[name, value] for name, value in zip(metric_names, metric_values)]

    table = tabulate(metric_rows, headers=["Metric", "Value"], tablefmt="grid")
    print(table)


def save_feature_rankings(feature_rankings, filename):
    lines = []
    for i, feature in enumerate(feature_rankings, start=1):
        lines.append(f"Feature {i}:")
        for j, ranking in enumerate(feature, start=1):
            frequency = (ranking / sum(feature)) * 100.
            lines.append(f"\tRank {j}: {ranking}/{sum(feature)} ({frequency:.0f}%)")

    rankings_file = open(filename, "w")
    rankings_file.write("\n".join(line for line in lines))

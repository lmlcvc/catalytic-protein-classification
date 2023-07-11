import os

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tabulate import tabulate

import tensorflow as tf
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

model_dir = config['model_dir']

node_feature_names = ["Residue name",
                      "Residue number",
                      "B-factor",
                      "X coordinate",
                      "Y coordinate",
                      "Z coordinate"]


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


def calculate_node_saliency(node_gradients):
    saliency_map = np.abs(node_gradients.numpy())
    saliency_map /= np.max(saliency_map)
    saliency_map = np.transpose(saliency_map)

    return saliency_map


def calculate_edge_saliency(edge_gradients):
    edge_saliency_map = np.abs(edge_gradients.numpy())
    edge_saliency_map /= np.max(edge_saliency_map)
    edge_saliency_map = np.squeeze(edge_saliency_map)

    return edge_saliency_map


def visualize_node_heatmap(heatmap, filename, figsize=(10, 8), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', aspect='auto')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Saliency')
    plt.xlabel('Node Index')
    plt.ylabel('Feature Index')
    plt.yticks(range(len(node_feature_names)), node_feature_names)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def visualize_edge_heatmap(heatmap, filename, figsize=(10, 8), dpi=300):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', aspect='equal')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Saliency')
    plt.xlabel('Source Node Index')
    plt.ylabel('Target Node Index')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def visualize_training(histories, figsize=(10, 6), dpi=300):
    # Plot loss
    plt.figure(figsize=figsize, dpi=dpi)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f"Fold {i + 1}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'training_loss.png'))
    plt.close()

    # Plot accuracy
    plt.figure(figsize=figsize, dpi=dpi)
    for i, history in enumerate(histories):
        plt.plot(history.history['acc'], label=f"Fold {i + 1}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'training_accuracy.png'))
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
    for i, feature in enumerate(feature_rankings):
        lines.append(f"{node_feature_names[i]}:")
        for j, ranking in enumerate(feature, start=1):
            frequency = (ranking / sum(feature)) * 100.
            lines.append(f"\tRank {j}: {ranking}/{sum(feature)} ({frequency:.0f}%)")

    rankings_file = open(filename, "w")
    rankings_file.write("\n".join(line for line in lines))

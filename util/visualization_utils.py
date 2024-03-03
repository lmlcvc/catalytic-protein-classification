import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tabulate import tabulate

import tensorflow as tf
import configparser
import seaborn as sns

import util.file_utils as fu

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

model_dir = config['model_dir']

node_feature_names = ["Residue name",
                      "B-factor",
                      "X coordinate",
                      "Y coordinate",
                      "Z coordinate"]


def plot_gradients(df, mode, output_dir, as_df=None, n=20):
    """
    Plot how many nodes pass gradient thresholds from a range of n elements.

    Args:
        df: sorted dataframe matching gradient values to row index (index in protein)
        mode: one of 'node', 'edge'
        output_dir: dir path to save the figure
        as_df: dataframe containing gradient info only for confirmed active site nodes
        n: number of thresholds in range

    Returns:

    """

    total_nodes = len(df)
    max_gradient = df.iloc[0]['gradient']
    min_gradient = df.iloc[-1]['gradient']

    # Create a range of gradient values
    gradient_range = np.linspace(min_gradient, max_gradient, num=n)

    # Find the insertion point for each threshold in the sorted gradient column
    counts = []
    for threshold in gradient_range:
        count = (df['gradient'] >= threshold).astype(int).sum()
        percentage = (count / total_nodes) * 100
        counts.append(percentage)

    # Plot the gradient counts
    plt.plot(gradient_range, counts, marker='o', label = 'overall')

    # Plot the gradient counts for as_df if provided
    if as_df is not None:
        total_as_nodes = len(as_df)
        as_counts = []
        for threshold in gradient_range:
            count = (as_df['gradient'] >= threshold).astype(int).sum()
            percentage = (count / total_as_nodes) * 100
            as_counts.append(percentage)
        plt.plot(gradient_range, as_counts, marker='o', color='red', label='active site')

    plt.xlabel('Threshold')
    plt.ylabel('Rows [%]')
    plt.title(f'Proportion of {mode}s over gradient threshold')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'gradient_thresholds_{mode}.png'))


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
    # plt.show()
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


def calculate_prediction_counts(predictions, truth_labels, neg_category_count=12, pos_category_count=8):
    # TODO: threshold could be not hardcoded
    # then calc category counts with parameters of threshold & range
    non_catalytic_predictions = []
    catalytic_predictions = []

    non_catalytic_false_counts = np.zeros(neg_category_count)
    catalytic_false_counts = np.zeros(pos_category_count)

    for prediction, truth_label in zip(predictions, truth_labels):
        category_value = int(prediction * 10) / 10
        if 0 <= prediction < 0.6:
            if int(truth_label) != 0:
                non_catalytic_false_counts[int(category_value * neg_category_count)] += 1
            else:
                non_catalytic_predictions.append(category_value)
        elif 0.6 <= prediction <= 1:
            if int(truth_label) != 1:
                catalytic_false_counts[int((category_value - 0.6) * pos_category_count)] += 1
            else:
                catalytic_predictions.append(category_value)
        else:
            raise ValueError(f"Prediction must be in [0, 1]. Was {prediction}")

    return non_catalytic_predictions, catalytic_predictions, non_catalytic_false_counts, catalytic_false_counts


def visualise_predictions(predictions, truth_labels, output_dir, category_count=10):
    fu.create_folder(output_dir)

    non_catalytic_predictions, \
        catalytic_predictions, \
        non_catalytic_false_counts, \
        catalytic_false_counts = calculate_prediction_counts(predictions, truth_labels, category_count)

    total_categories = category_count * 2
    category_width = 1.0 / total_categories

    non_catalytic_x = np.arange(0, 0.6 + category_width, category_width)
    catalytic_x = np.arange(0.6, 1 + category_width, category_width)

    plt.figure(figsize=(8, 6))
    plt.bar(non_catalytic_x[:-1], np.histogram(non_catalytic_predictions, bins=non_catalytic_x)[0],
            width=0.9 * category_width, align='edge', label='True', color='blue')

    # FIXME: ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (12,) and arg 1 with shape (10,).
    plt.bar(non_catalytic_x[:-1], non_catalytic_false_counts,
            width=0.9 * category_width, align='edge',
            bottom=np.histogram(non_catalytic_predictions, bins=non_catalytic_x)[0],
            label='False', color='red', alpha=0.7)

    plt.xlabel('Prediction Category')
    plt.ylabel('Count')
    plt.title('Distribution of Non-catalytic Predictions')
    plt.xticks(non_catalytic_x)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'non_catalytic_predictions_histogram'), bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.bar(catalytic_x[:-1], np.histogram(catalytic_predictions, bins=catalytic_x)[0],
            width=0.9 * category_width, align='edge', label='True', color='blue')

    plt.bar(catalytic_x[:-1], catalytic_false_counts,
            width=0.9 * category_width, align='edge', bottom=np.histogram(catalytic_predictions, bins=catalytic_x)[0],
            label='False', color='red', alpha=0.7)

    plt.xlabel('Prediction Category')
    plt.ylabel('Count')
    plt.title('Distribution of Catalytic Predictions')
    plt.xticks(catalytic_x)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'catalytic_predictions_histogram'), bbox_inches='tight')


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
        plt.plot(history.history['binary_accuracy'], label=f"Fold {i + 1}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'training_accuracy.png'))
    plt.close()

    plt.figure(figsize=figsize, dpi=dpi)
    for i, history in enumerate(histories):
        plt.plot(history.history['val_binary_accuracy'], label=f"Fold {i + 1}")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'validation_accuracy.png'))
    plt.close()


def evaluate_model(predictions, labels, both_classes_present=True):
    predictions = [prediction for sublist in predictions for prediction in sublist]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions) if both_classes_present else -1

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


def feature_correlations(log_path, out_dir):
    """
    TODO: more detail
    Analyse correlations between important features.
    See if there are features that tend to be important together.
    This might indicate that certain sets of features are more informative for the model.
    """
    df = pd.read_csv(log_path)
    correlation_matrix = df.corr()

    # Visualize the correlation matrix as a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlations')
    plt.savefig(os.path.join(out_dir, 'correlation.png'))

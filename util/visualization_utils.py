import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, \
    confusion_matrix
from tabulate import tabulate

import util.file_utils as fu

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

model_dir = config['model_dir']
graph_type = config['graph_type']

if graph_type == 'molecule':
    node_feature_names = ["Atomic Number",
                          "Atomic Mass",
                          "Degree",
                          "Total degree",
                          "Total valence",
                          "Explicit valence",
                          # "Implicit valence",
                          # "Number of explicit H",
                          # "Number of implicit H",
                          "Total number of H",
                          "Number of radical electrons",
                          "Formal charge",
                          "Hybridization",
                          "Is Aromatic",
                          # "Is Isotope",
                          "Is Ring",
                          "Chiral tag",
                          "Is C",
                          "Is H",
                          "Is O",
                          "Is N",
                          "Is F",
                          "Is P",
                          "Is S",
                          "Is Cl",
                          "Is Br",
                          "Is I",
                          "Is B", ]
else:
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


def calculate_prediction_counts(predictions, truth_labels, category_count):
    non_catalytic_predictions = []
    catalytic_predictions = []

    non_catalytic_false_counts = np.zeros(category_count)
    catalytic_false_counts = np.zeros(category_count)

    for prediction, truth_label in zip(predictions, truth_labels):
        category_value = int(prediction * 10) / 10
        if 0 <= prediction < 0.5:
            if np.round(prediction) != truth_label:
                non_catalytic_false_counts[int(category_value * category_count)] += 1
            else:
                non_catalytic_predictions.append(category_value)
        elif 0.5 <= prediction <= 1:
            if np.round(prediction) != truth_label:
                catalytic_false_counts[int((category_value - 0.5) * category_count)] += 1
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

    non_catalytic_x = np.arange(0, 0.5 + category_width, category_width)
    catalytic_x = np.arange(0.5, 1 + category_width, category_width)

    plt.figure(figsize=(8, 6))
    plt.bar(non_catalytic_x[:-1], np.histogram(non_catalytic_predictions, bins=non_catalytic_x)[0],
            width=0.9 * category_width, align='edge', label='True', color='blue')

    plt.bar(non_catalytic_x[:-1], non_catalytic_false_counts,
            width=0.9 * category_width, align='edge',
            bottom=np.histogram(non_catalytic_predictions, bins=non_catalytic_x)[0],
            label='False', color='red', alpha=0.7)

    plt.xlabel('Prediction Category')
    plt.ylabel('Count')
    plt.title('Distribution of Negative Class Predictions')
    plt.xticks(non_catalytic_x)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'negative_predictions_histogram'), bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    plt.bar(catalytic_x[:-1], np.histogram(catalytic_predictions, bins=catalytic_x)[0],
            width=0.9 * category_width, align='edge', label='True', color='blue')

    plt.bar(catalytic_x[:-1], catalytic_false_counts,
            width=0.9 * category_width, align='edge', bottom=np.histogram(catalytic_predictions, bins=catalytic_x)[0],
            label='False', color='red', alpha=0.7)

    plt.xlabel('Prediction Category')
    plt.ylabel('Count')
    plt.title('Distribution of Positive Class Predictions')
    plt.xticks(catalytic_x)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'positive_predictions_histogram'), bbox_inches='tight')


def visualize_roc(predictions, truth_labels, output_dir):
    fpr, tpr, thresholds = roc_curve(truth_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))


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


def visualize_validation(histories, figsize=(10, 6), dpi=300):
    # Plot loss
    plt.figure(figsize=figsize, dpi=dpi)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label="Training")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(os.path.join(model_dir, f'training_validation_loss_fold_{i + 1}.png'))
        plt.close()

    # Plot accuracy
    plt.figure(figsize=figsize, dpi=dpi)
    for i, history in enumerate(histories):
        plt.plot(history.history['acc'], label="Training")
        plt.plot(history.history['val_acc'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.savefig(os.path.join(model_dir, f'training_validation_accuracy_fold_{i + 1}.png'))
        plt.close()


def evaluate_model(predictions, labels):
    predictions = [prediction for sublist in predictions for prediction in sublist]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    fpr = fp / (fp + tn)

    metric_names = ["Accuracy", "Precision", "Recall", "False positive rate", "F1-score", "ROC AUC"]
    metric_values = [accuracy, precision, recall, fpr, f1, roc_auc]

    metric_rows = [[name, value] for name, value in zip(metric_names, metric_values)]

    table = tabulate(metric_rows, headers=["Metric", "Value"], tablefmt="grid")
    print(table)
    return metric_values


def visualize_multiple_models(metrics, figsize=(10, 6), dpi=300):
    metric_names = ["Accuracy", "Precision", "Recall", "False positive rate", "F1-score", "ROC AUC"]
    # Boxplot
    plt.figure(figsize=figsize, dpi=dpi)
    melted_df = metrics.reset_index().melt(id_vars='index',
                                           var_name='Metric',
                                           value_name='Score')
    ax = sns.boxplot(
        x="Metric",
        y="Score",
        data=melted_df,
        palette="viridis",
        linewidth=1.5,
        flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'}
    )

    plt.title("Cross-Validation Metrics Distribution Across Folds", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)  # Adjust if metrics exceed 1.0
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add median values as text
    medians = melted_df.groupby("Metric")['Score'].median().round(3)
    for i, metric in enumerate(metric_names):
        ax.text(i, medians[metric] + 0.02, f'Med: {medians[metric]}',
                ha='center', color='black', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'box_plot.png'))
    plt.close()

    # Radar chart
    categories = metrics.columns.tolist()
    num_vars = len(categories)

    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Plot setup
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, polar=True)

    # Plot each row
    for idx, row in metrics.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]  # Close the line
        ax.plot(angles, values, linewidth=1, label=row.name)
        ax.fill(angles, values, alpha=0.1)

    # Formatting
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_rlabel_position(0)
    ax.tick_params(axis='both', which='major', pad=10)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"Fold {i + 1}" for i in range(len(labels))]
    plt.legend(handles=handles, labels=new_labels, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Metric Comparison Radar Chart", y=1.1)
    plt.savefig(os.path.join(model_dir, 'radar_chart.png'))
    plt.close()

    # Bar graph
    plt.figure(figsize=figsize, dpi=dpi)
    sns.barplot(x='Metric', y='Score', hue='index', data=melted_df, palette='viridis')

    # Formatting
    plt.title("Metric Comparison - Grouped Bar Chart")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(handles=handles, labels=new_labels, title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'bar_chart.png'))
    plt.close()


def save_feature_rankings(feature_rankings, filename):
    lines = []
    for i, feature in enumerate(feature_rankings):
        lines.append(f"{node_feature_names[i]}:")
        for j, ranking in enumerate(feature, start=1):
            frequency = (ranking / sum(feature)) * 100.
            lines.append(f"\tRank {j}: {ranking}/{sum(feature)} ({frequency:.0f}%)")

    rankings_file = open(filename, "w")
    rankings_file.write("\n".join(line for line in lines))

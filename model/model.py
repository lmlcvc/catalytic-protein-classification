from keras import Model, Input
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph.layer import DeepGraphCNN
import matplotlib.pyplot as plt
import numpy as np
import os
from tabulate import tabulate

import tensorflow as tf


def in_out_tensors(generator, model):
    """
    Builds a Graph Classification model.

    Returns:
        tuple: ``(x_inp, x_out)``, where ``x_inp`` is a list of two input tensors for the
            Graph Classification model (containing node features and normalized adjacency matrix),
            and ``x_out`` is a tensor for the Graph Classification model output.
    """
    x_t = Input(shape=(None, generator.node_features_size))
    mask = Input(shape=(None,), dtype=tf.bool)
    A_m = Input(shape=(None, None))

    x_inp = [x_t, mask, A_m]
    x_out = model(x_inp)

    return x_inp, x_out


def create_graph_classification_model_gcn(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = in_out_tensors(generator, gc_model)
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model


def create_graph_classification_model_dcgnn(generator):
    k = 35  # the number of rows for the output tensor
    layer_sizes = [32, 32, 32, 1]

    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh"],
        k=k,
        bias=False,
        generator=generator,
    )
    x_inp, x_out = in_out_tensors(generator, dgcnn_model)

    x_out = tf.keras.layers.Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = tf.keras.layers.MaxPool1D(pool_size=2)(x_out)

    x_out = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = tf.keras.layers.Flatten()(x_out)

    x_out = tf.keras.layers.Dense(units=128, activation="relu")(x_out)
    x_out = tf.keras.layers.Dropout(rate=0.5)(x_out)

    predictions = tf.keras.layers.Dense(units=1, activation="sigmoid")(x_out)

    # Create the Keras model and prepare it for training
    model = tf.keras.Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["acc"])

    return model


# Create a function to get the gradients of the output predictions with respect to the input graph nodes
@tf.function
def get_gradients(model, inputs):
    x_t, mask, A_m = inputs  # Unpack the input tensors

    mask_float = tf.where(mask, tf.ones_like(mask, dtype=tf.float32),
                          tf.zeros_like(mask, dtype=tf.float32))

    inputs_adapted = [x_t, mask_float, A_m]

    with tf.GradientTape() as tape:
        tape.watch(inputs[0])
        predictions = model(inputs)

    # gradients = tape.gradient(predictions, inputs_adapted)
    gradients = tape.gradient(predictions, inputs[0])

    return gradients


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

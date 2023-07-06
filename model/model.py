from keras import Model
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn import model_selection
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph.layer import DeepGraphCNN
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf


def create_graph_classification_model_gcn(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )

    # TODO: Napravit se bezobrazan i napisati svoju in_out_tensors()
    x_inp, x_out = gc_model.in_out_tensors()
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
    x_inp, x_out = dgcnn_model.in_out_tensors()

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
    inputs = tf.cast(inputs, tf.float32)  # Cast the inputs to float32
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
    gradients = tape.gradient(predictions, inputs)
    return gradients


def visualize_grad_cam(heatmap, filename):
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph.layer import DeepGraphCNN

import tensorflow as tf


def create_graph_classification_model_gcn(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()

    predictions = tf.keras.layers.Dense(units=32, activation="relu")(x_out)
    predictions = tf.keras.layers.Dense(units=16, activation="relu")(predictions)
    predictions = tf.keras.layers.Dense(units=1, activation="sigmoid")(predictions)

    # Create the Keras model and prepare it for training
    model = tf.keras.Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["acc"])

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

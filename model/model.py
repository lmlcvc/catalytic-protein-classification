import tensorflow as tf
from stellargraph.layer import DeepGraphCNN
from stellargraph.layer import GCNSupervisedGraphClassification
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam


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
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.001),
                  loss=binary_crossentropy,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

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

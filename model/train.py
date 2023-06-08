import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn import model_selection

from model.model import create_graph_classification_model_gcn

# from tensorflow.tools.api import generator

# from tests.layer.test_graph_classification import generator

# epochs = 200  # maximum number of training epochs
# folds = 10  # the number of folds for k-fold cross validation
# n_repeats = 5  # the number of repeats for repeated k-fold cross validation

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)


def train_fold(model, train_gen, test_gen, es, epochs):
    print(f"train gen: {train_gen}")
    print(f"validation data (test gen):{test_gen}")
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc


def get_generators(generator, train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


def train_model(graph_generator, graph_labels, epochs=200, folds=10, n_repeats=5):
    """
    :param epochs: maximum number of training epochs
    :param folds: the number of folds for k-fold cross validation
    :param n_repeats: the number of repeats for repeated k-fold cross validation
    :param graph_generator: 
    :param graph_labels:
    :return:
    """

    test_accs = []

    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)

    for i, (train_index, test_index) in enumerate(stratified_folds):
        print(f"Training and evaluating on fold {i + 1} out of {folds * n_repeats}...")
        train_gen, test_gen = get_generators(
            graph_generator, train_index, test_index, graph_labels, batch_size=30
        )

        model = create_graph_classification_model_gcn(graph_generator)

        history, acc = train_fold(model, train_gen, test_gen, es, epochs)

        test_accs.append(acc)

    print(
        f"Accuracy over all folds mean: {np.mean(test_accs) * 100:.3}% and std: {np.std(test_accs) * 100:.2}%"
    )

    plt.figure(figsize=(8, 6))
    plt.hist(test_accs)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")

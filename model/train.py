import configparser

import numpy as np
from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping
from sklearn import model_selection

import util.visualization_utils as vu

from model.model import create_graph_classification_model_gcn, create_graph_classification_model_dcgnn

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

use_dgcnn = config['use_dgcnn']

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)


def train_fold(model, train_gen, test_gen, es, epochs):
    print(f"train gen: {train_gen}")
    print(f"validation data (test gen):{test_gen}")
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=1, callbacks=[es],
    )

    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=1)
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
    test_accs = []
    all_histories = []
    best_model = None
    best_acc = 0.

    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)

    for i, (train_index, test_index) in enumerate(stratified_folds):
        print(f"Training and evaluating on fold {i + 1} out of {folds * n_repeats}...")
        train_gen, test_gen = get_generators(
            graph_generator, train_index, test_index, graph_labels, batch_size=8
        )

        if use_dgcnn.lower() == "y":
            model = create_graph_classification_model_dcgnn(graph_generator)
        else:
            model = create_graph_classification_model_gcn(graph_generator)

        history, acc = train_fold(model, train_gen, test_gen, es, epochs)
        all_histories.append(history)
        test_accs.append(acc)

        print(f"Train set size: {len(train_index)} graphs")
        print(f"Test set size: {len(test_index)} graphs")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print(
        f"Accuracy over all folds mean: {np.mean(test_accs) * 100:.3}% and std: {np.std(test_accs) * 100:.2}%"
    )

    vu.visualize_training(all_histories)

    plt.figure(figsize=(8, 6))
    plt.hist(test_accs)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.show()

    return best_model

import configparser
import csv
import os

import numpy as np
from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping
from sklearn import model_selection
from datetime import datetime

import util.visualization_utils as vu

from model.model import create_graph_classification_model_gcn, create_graph_classification_model_dcgnn

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

use_dgcnn = config['use_dgcnn']
model_dir = config['model_dir']

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)


def train_fold(model, train_gen, test_gen, es, epochs):
    print(f"train gen: {train_gen}")
    print(f"validation data (test gen):{test_gen}")

    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=1, callbacks=[es],
    )

    # Accessing metrics
    metrics = {"train_acc": history.history['acc'],
               "val_acc": history.history['val_acc'],
               "train_precision": history.history['precision'],
               "val_precision": history.history['val_precision'],
               "train_recall": history.history['recall'],
               "val_recall": history.history['val_recall'],
               "train_f1": history.history['f1'],
               "val_f1": history.history['val_f1']}

    # calculate performance on the test data and return along with history
    # test_metrics = model.evaluate(test_gen, verbose=1)
    # test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, metrics


def get_generators(generator, train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


def log_metrics_to_file(metrics, file_path, fold):
    is_file_empty = not os.path.isfile(file_path) or os.path.getsize(file_path) == 0

    with open(file_path, "a", newline='') as csvfile:
        fieldnames = list(metrics.keys()) + ['fold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not is_file_empty:
            writer.writeheader()

        metrics['fold'] = fold
        writer.writerow(metrics)


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
            graph_generator, train_index, test_index, graph_labels, batch_size=5
        )

        if use_dgcnn.lower() == "y":
            model = create_graph_classification_model_dcgnn(graph_generator)
        else:
            model = create_graph_classification_model_gcn(graph_generator)

        history, metrics = train_fold(model, train_gen, test_gen, es, epochs)
        all_histories.append(history)
        test_accs.append(metrics["val_acc"])

        run_timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        log_metrics_to_file(metrics, os.path.join(model_dir, f"{run_timestamp}.csv"), fold=i + 1)

        print(f"Train set size: {len(train_index)} graphs")
        print(f"Test set size: {len(test_index)} graphs")

        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
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

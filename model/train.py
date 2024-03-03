import configparser
import csv
import os

import pandas as pd
import tensorflow.keras as keras
import numpy as np
from matplotlib import pyplot as plt

from sklearn import model_selection
from datetime import datetime

import util.analysis_utils as au
import util.visualization_utils as vu

from model.model import create_graph_classification_model_gcn, create_graph_classification_model_dcgnn

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

use_dgcnn = config['use_dgcnn']
model_dir = config['model_dir']


def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=1, callbacks=[es],
    )

    # Accessing metrics
    metrics = {"train_acc": history.history['binary_accuracy'],
               "val_acc": history.history['val_binary_accuracy'],
               "train_precision": history.history['precision'],
               "val_precision": history.history['val_precision'],
               "train_recall": history.history['recall'],
               "val_recall": history.history['val_recall']}

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
        fieldnames = list(metrics.keys()) + ['fold', 'epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if is_file_empty:
            writer.writeheader()

        # Get the length of the lists in metrics (assuming all lists have the same length)
        num_rows = len(next(iter(metrics.values())))

        for i in range(num_rows):
            # Construct a row dictionary with the i-th element of each list
            row = {key: round(metrics[key][i], 2) for key in metrics}
            row['fold'] = fold
            row['epoch'] = i + 1
            writer.writerow(row)


def train_model(graph_generator, graph_labels, run_dir, training_tensors, epochs=200, folds=10, n_repeats=5):
    test_accs = []
    all_histories = []
    best_model = None
    best_acc = -1.

    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
    )

    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
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
        test_accs.append(max(metrics["val_acc"]))  # TODO: double check this max

        log_metrics_to_file(metrics, os.path.join(model_dir, f"{run_timestamp}.csv"), fold=i + 1)

        print(f"Train set size: {len(train_index)} graphs")
        print(f"Test set size: {len(test_index)} graphs")

        node_dataframes = []
        edge_dataframes = []
        for idx, graph in enumerate(test_index):
            protein = graph_labels.index[idx]
            inputs = training_tensors[idx][0]

            gradients = vu.get_gradients(model, inputs)
            node_gradients = gradients[0]
            edge_gradients = gradients[-1]

            node_dataframes.append(au.extract_relevant_gradients(protein, node_gradients))
            edge_dataframes.append(au.extract_relevant_gradients(protein, edge_gradients))

            # Saliency maps
            node_saliency_map = vu.calculate_node_saliency(gradients[0])
            edge_saliency_map = vu.calculate_edge_saliency(gradients[-1])

            # Visualize the saliency maps and save them as images
            vu.visualize_node_heatmap(node_saliency_map, os.path.join(run_dir, f"node_saliency_map-{i}.png"))
            vu.visualize_edge_heatmap(edge_saliency_map, os.path.join(run_dir, f"edge_saliency_map-{i}.png"))

        most_relevant_nodes = pd.concat(node_dataframes, ignore_index=True)
        most_relevant_nodes_sorted = most_relevant_nodes.sort_values(by='gradient', ascending=False)
        active_site_nodes = au.filter_active_site_gradients(most_relevant_nodes_sorted)

        most_relevant_edges = pd.concat(edge_dataframes, ignore_index=True)
        most_relevant_edges_sorted = most_relevant_edges.sort_values(by='gradient', ascending=False)
        active_site_edges = au.filter_active_site_gradients(most_relevant_edges_sorted)

        vu.plot_gradients(most_relevant_nodes_sorted, mode='node', output_dir=run_dir, as_df=active_site_nodes)
        vu.plot_gradients(most_relevant_edges_sorted, mode='edge', output_dir=run_dir, as_df=active_site_edges)

        if max(metrics["val_acc"]) > best_acc:
            best_acc = max(metrics["val_acc"])
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

import configparser
import os

import numpy as np
import pickle
from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import pickle
import os
import tensorflow as tf

import util.visualization_utils as vu
import util.file_utils as fu

from model.model import create_graph_classification_model_gcn, create_graph_classification_model_dgcnn

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

use_dgcnn = config['use_dgcnn']

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)


def train_fold(model, train_gen, test_gen, es, epochs, class_weights):
    print(f"train gen: {train_gen}")
    print(f"validation data (test gen):{test_gen}")
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=1, callbacks=[es], class_weight=class_weights
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


def generate_fold_indices(graph_labels, split_dir, folds=10, n_repeats=1):
    """
    Generate file with fold_num lists of training, validation and test indices
    """

    fu.create_folder(split_dir)

    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)

    all_splits = {}
    for i, (train_index, val_index) in enumerate(stratified_folds):
        all_splits[i] = {
            'train': train_index,
            'val': val_index
        }

    with open(os.path.join(split_dir, 'fold_indices.pkl'), 'wb') as f:
        pickle.dump(all_splits, f)
    print(f"Saved {len(all_splits)} folds to fold_indices.pkl")


def train_model(graph_generator, graph_labels, class_weights, epochs=200, folds=10, n_repeats=5):
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
            model = create_graph_classification_model_dgcnn(graph_generator)
        else:
            model = create_graph_classification_model_gcn(graph_generator)

        history, acc = train_fold(model, train_gen, test_gen, es, epochs, class_weights)
        all_histories.append(history)
        test_accs.append(acc)

        print(f"Train set size: {len(train_index)} graphs")
        print(f"Validation set size: {len(test_index)} graphs")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print(
        f"Accuracy over all folds mean: {np.mean(test_accs) * 100:.3}% and std: {np.std(test_accs) * 100:.2}%"
    )

    vu.visualize_training(all_histories)
    vu.visualize_validation(all_histories)

    plt.figure(figsize=(8, 6))
    plt.hist(test_accs)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.show()

    return best_model


def train_fold_single(graph_generator, graph_labels, class_weights, split_dir, fold_num, epochs=200):
    with open(os.path.join(split_dir, 'fold_indices.pkl'), 'rb') as f:
        all_splits = pickle.load(f)

    train_index = all_splits[fold_num]['train']
    val_index = all_splits[fold_num]['val']

    return train_model_single(graph_generator, graph_labels, class_weights, train_index, val_index, epochs)


def train_model_single(graph_generator, graph_labels, class_weights, train_index, val_index, epochs=200):
    train_gen, test_gen = get_generators(
        graph_generator, train_index, val_index, graph_labels, batch_size=8
    )

    if use_dgcnn.lower() == "y":
        model = create_graph_classification_model_dgcnn(graph_generator)
    else:
        model = create_graph_classification_model_gcn(graph_generator)

    history, acc = train_fold(model, train_gen, test_gen, es, epochs, class_weights)

    print(f"Train set size: {len(train_index)} graphs")
    print(f"Validation set size: {len(val_index)} graphs")
    print(f"Accuracy on validation set: {acc}%")

    return model, history


def perform_benchmark(model_dir, split_dir, graph_generator, graph_labels, fold, model):
    """
    For a single fold, use the provided loaded model to extract embeddings, train SVM and RF, and save results.
    Args:
        model_dir (str): directory where per-fold Keras models are stored (model_{fold}.h5)
        split_dir (str): directory where fold indices file 'fold_indices.pkl' is stored
        graph_generator: PaddedGraphGenerator instance
        graph_labels: pandas Series of labels
        fold (int): fold number to run
        model: loaded Keras model for this fold
    """
    bench_dir = os.path.join(model_dir, 'benchmarks')
    os.makedirs(bench_dir, exist_ok=True)

    split_file = os.path.join(split_dir, 'fold_indices.pkl')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Fold indices not found at {split_file}. Generate with --generate-folds")

    with open(split_file, 'rb') as f:
        all_splits = pickle.load(f)

    if fold not in all_splits:
        print(f"Fold {fold} not found in split indices.")
        return

    print(f"Benchmarking fold {fold}")

    embed_layer = model.get_layer('flatten_embedding')
    embedding_model = tf.keras.Model(inputs=model.input, outputs=embed_layer.output)

    train_idx = all_splits[fold]['train']
    val_idx = all_splits[fold]['val']

    def compute_embeddings(indices):
        gen = graph_generator.flow(indices, targets=None, batch_size=8, shuffle=False)
        emb = embedding_model.predict(gen, verbose=0)
        return emb

    X_train = compute_embeddings(train_idx)
    X_val = compute_embeddings(val_idx)
    y_train = graph_labels.iloc[train_idx].values
    y_val = graph_labels.iloc[val_idx].values

    svm = make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf', C=1.0))
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    print('Training SVM...')
    svm.fit(X_train, y_train)
    print('Training RandomForest...')
    rf.fit(X_train, y_train)

    svm_acc = svm.score(X_val, y_val)
    rf_acc = rf.score(X_val, y_val)
    print(f'Fold {fold} validation accuracies - SVM: {svm_acc:.4f}, RF: {rf_acc:.4f}')

    joblib.dump(svm, os.path.join(bench_dir, f'svm_fold_{fold}.joblib'))
    joblib.dump(rf, os.path.join(bench_dir, f'rf_fold_{fold}.joblib'))
    with open(os.path.join(bench_dir, f'meta_fold_{fold}.pkl'), 'wb') as mf:
        pickle.dump({'svm_val_acc': float(svm_acc), 'rf_val_acc': float(rf_acc)}, mf)

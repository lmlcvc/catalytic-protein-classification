import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn import model_selection
from sklearn.metrics import accuracy_score

from model.model import create_graph_classification_model_gcn

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
    test_accs = []

    stratified_folds = model_selection.RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=n_repeats
    ).split(graph_labels, graph_labels)

    model = create_graph_classification_model_gcn(graph_generator)  # Initialize the model

    for i, (train_index, test_index) in enumerate(stratified_folds):
        print(f"Training and evaluating on fold {i + 1} out of {folds * n_repeats}...")
        train_gen, test_gen = get_generators(
            graph_generator, train_index, test_index, graph_labels, batch_size=30
        )

        history, acc = train_fold(model, train_gen, test_gen, es, epochs)
        test_accs.append(acc)

        print(f"Train set size: {len(train_index)} graphs")
        print(f"Test set size: {len(test_index)} graphs")

    print(
        f"Accuracy over all folds mean: {np.mean(test_accs) * 100:.3}% and std: {np.std(test_accs) * 100:.2}%"
    )

    plt.figure(figsize=(8, 6))
    plt.hist(test_accs)
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.show()

    return model
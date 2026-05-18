import csv
import os
import random

import numpy as np
from matplotlib import pyplot as plt

from metrics import Metrics
from utilities import sigmoid, sigmoid_derivative_from_output, mse, ensure_dir


class MLP:
    def __init__(
            self,
            layers,
            learning_rate=0.6,
            momentum=0.0,
            use_bias=True,
            use_momentum=False,
            weight_init_min=-0.5,
            weight_init_max=0.5,
            seed=42
    ):

        random.seed(seed)
        np.random.seed(seed)

        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_bias = use_bias
        self.use_momentum = use_momentum

        self.weights = []
        self.previous_updates = []

        self.outputs = []
        self.deltas = []

        self._initialize_weights(weight_init_min, weight_init_max)
        self.epoch_target = 0

    # ==========================================================
    # INIT
    # ==========================================================

    def _initialize_weights(self, wmin, wmax):
        for layer_index in range(len(self.layers) - 1):
            input_size = self.layers[layer_index]
            output_size = self.layers[layer_index + 1]

            if self.use_bias:
                input_size += 1

            count = input_size * output_size

            w = np.array(
                [random.uniform(wmin, wmax) for _ in range(count)],
                dtype=float
            )

            prev = np.zeros(count, dtype=float)

            self.weights.append(w)
            self.previous_updates.append(prev)

    # ==========================================================
    # FORWARD
    # ==========================================================

    def forward(self, inputs, return_all_layers=False):
        self.outputs = [inputs[:]]
        current = inputs[:]

        for layer_index, weight_vector in enumerate(self.weights):
            input_size = self.layers[layer_index]
            output_size = self.layers[layer_index + 1]

            if self.use_bias:
                current_with_bias = current + [1.0]
                effective_input_size = input_size + 1
            else:
                current_with_bias = current
                effective_input_size = input_size

            next_outputs = []

            for neuron in range(output_size):
                base = neuron * effective_input_size
                w = weight_vector[base: base + effective_input_size]

                s = 0.0
                for wi, xi in zip(w, current_with_bias):
                    s += wi * xi

                next_outputs.append(sigmoid(s))

            self.outputs.append(next_outputs)
            current = next_outputs

        return (current, self.outputs) if return_all_layers else current

    # ==========================================================
    # BACKPROP
    # ==========================================================

    def backward(self, targets):
        self.deltas = [None] * len(self.layers)

        output_layer = len(self.layers) - 1

        outputs = self.outputs[-1]
        output_deltas = []

        for o, t in zip(outputs, targets):
            error = t - o
            delta = error * sigmoid_derivative_from_output(o)
            output_deltas.append(delta)

        self.deltas[output_layer] = output_deltas

        # hidden layers
        for layer in range(output_layer - 1, 0, -1):

            current_deltas = []
            current_outputs = self.outputs[layer]
            next_deltas = self.deltas[layer + 1]

            weights = self.weights[layer]

            current_size = self.layers[layer]
            next_size = self.layers[layer + 1]

            effective_current_size = current_size + 1 if self.use_bias else current_size

            for neuron in range(current_size):

                error_sum = 0.0

                for next_neuron in range(next_size):
                    idx = next_neuron * effective_current_size + neuron
                    error_sum += weights[idx] * next_deltas[next_neuron]

                delta = error_sum * sigmoid_derivative_from_output(current_outputs[neuron])
                current_deltas.append(delta)

            self.deltas[layer] = current_deltas

    # ==========================================================
    # WEIGHT UPDATE
    # ==========================================================

    def update_weights(self):
        for layer_index in range(len(self.weights)):

            inputs = self.outputs[layer_index]

            if self.use_bias:
                inputs = inputs + [1.0]

            deltas = self.deltas[layer_index + 1]

            input_size = len(inputs)
            output_size = len(deltas)

            for neuron in range(output_size):
                base = neuron * input_size

                for i in range(input_size):

                    grad = self.learning_rate * deltas[neuron] * inputs[i]
                    idx = base + i

                    if self.use_momentum:
                        update = grad + self.momentum * self.previous_updates[layer_index][idx]
                        self.weights[layer_index][idx] += update
                        self.previous_updates[layer_index][idx] = update
                    else:
                        self.weights[layer_index][idx] += grad

    # ==========================================================
    # EVALUATE
    # ==========================================================

    def evaluate(self, dataset):
        predictions = []
        total_error = 0.0

        for x, y in zip(dataset.X, dataset.Y):
            out = self.forward(x)
            predictions.append(out)
            total_error += mse(y, out)

        accuracy = Metrics.accuracy(dataset.Y, predictions)
        avg_error = total_error / max(1, len(dataset.X))

        return accuracy, avg_error

    # ==========================================================
    # TRAIN
    # ==========================================================

    def train(
            self,
            dataset,
            epochs=5000,
            target_error=0.001,
            shuffle=True,
            stop_on_target=True,
            patience=500,
            log_every=10,
            output_dir="outputs"
    ):

        ensure_dir(output_dir)

        training_csv = os.path.join(output_dir, "training_log.csv")

        history_epochs = []
        history_errors = []
        history_accuracy = []

        best_error = float("inf")
        epochs_without_improvement = 0

        with open(training_csv, "w", newline="") as csvfile:

            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "error", "accuracy"])

            for epoch in range(1, epochs + 1):

                indices = list(range(len(dataset.X)))

                if shuffle:
                    random.shuffle(indices)

                for idx in indices:
                    x = dataset.X[idx]
                    y = dataset.Y[idx]

                    self.forward(x)
                    self.backward(y)
                    self.update_weights()

                accuracy, evaluated_error = self.evaluate(dataset)

                history_epochs.append(epoch)
                history_errors.append(evaluated_error)
                history_accuracy.append(accuracy)

                if epoch % log_every == 0:
                    writer.writerow([epoch, evaluated_error, accuracy])
                    print(f"[TRAIN] epoch={epoch} error={evaluated_error:.6f} accuracy={accuracy:.4f}")

                if evaluated_error < best_error:
                    best_error = evaluated_error
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

                if stop_on_target and evaluated_error <= target_error:
                    print(f"Target reached at epoch {epoch}")
                    self.epoch_target = epoch
                    break

        self.epoch_target = epoch

        self.plot_errors(history_epochs, history_errors, output_dir)
        self.plot_accuracy(history_epochs, history_accuracy, output_dir)

        num_classes = len(self.forward(dataset.X[0]))

        cm = Metrics.confusion_matrix(
            dataset.Y,
            self.predict_all(dataset),
            num_classes
        )

        Metrics.save_confusion_matrix(cm, output_dir)

    # ==========================================================
    # TEST
    # ==========================================================

    def test(self, dataset, output_dir="outputs"):

        ensure_dir(output_dir)

        predictions = []
        total_error = 0.0

        all_true = []
        all_pred = []

        test_dump = os.path.join(output_dir, "test_results.txt")

        with open(test_dump, "w", encoding="utf-8") as f:

            for i in range(len(dataset.X)):

                x = dataset.X[i]
                y = dataset.Y[i]

                out = self.forward(x)

                predictions.append(out)

                err_vec = [t - o for t, o in zip(y, out)]

                mse_err = mse(y, out)

                total_error += mse_err

                all_true.append(y)
                all_pred.append(out)

                f.write("=" * 80 + "\n")
                f.write(f"SAMPLE {i}\n\n")

                f.write(f"INPUT:\n{x}\n\n")

                f.write(f"TARGET:\n{y}\n\n")

                f.write(f"OUTPUT:\n{out}\n\n")

                f.write(f"ERROR VECTOR:\n{err_vec}\n\n")

                f.write(f"MSE:\n{mse_err}\n\n")

                # ======================================================
                # HIDDEN OUTPUTS
                # ======================================================

                f.write("HIDDEN LAYER OUTPUTS\n")

                if len(self.outputs) > 2:

                    for hidden_index in range(1, len(self.outputs) - 1):
                        f.write(
                            f"Layer {hidden_index}: "
                            f"{self.outputs[hidden_index]}\n"
                        )

                else:
                    f.write("No hidden layers\n")

                f.write("\n")

                # ======================================================
                # WEIGHTS
                # ======================================================

                f.write("NETWORK WEIGHTS\n")

                for layer_index in range(
                        len(self.weights) - 1,
                        -1,
                        -1
                ):

                    f.write(
                        f"\nLAYER {layer_index}\n"
                    )

                    input_size = self.layers[layer_index]

                    if self.use_bias:
                        input_size += 1

                    output_size = self.layers[layer_index + 1]

                    for neuron_index in range(output_size):
                        base = neuron_index * input_size

                        neuron_weights = self.weights[
                                             layer_index
                                         ][base: base + input_size]

                        f.write(
                            f"Neuron {neuron_index}: "
                            f"{neuron_weights.tolist()}\n"
                        )

                f.write("\n")

        avg_error = total_error / max(1, len(dataset.X))

        accuracy = Metrics.accuracy(
            all_true,
            all_pred
        )

        num_classes = len(
            self.forward(dataset.X[0])
        )

        cm = Metrics.confusion_matrix(
            dataset.Y,
            predictions,
            num_classes
        )

        Metrics.save_confusion_matrix(
            cm,
            output_dir
        )

        precision, recall, f1 = (
            Metrics.precision_recall_f1(cm)
        )

        per_class_acc = (
            Metrics.per_class_accuracy(cm)
        )

        with open(
                os.path.join(output_dir, "metrics.txt"),
                "w"
        ) as f:

            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Avg MSE: {avg_error}\n\n")

            f.write(
                "Precision:\n"
                + str(precision)
                + "\n"
            )

            f.write(
                "Recall:\n"
                + str(recall)
                + "\n"
            )

            f.write(
                "F1:\n"
                + str(f1)
                + "\n"
            )

            f.write(
                "Per-class accuracy:\n"
                + str(per_class_acc)
                + "\n"
            )

    # ==========================================================
    # UTIL
    # ==========================================================

    def predict_all(self, dataset):
        return [self.forward(x) for x in dataset.X]

    # ==========================================================
    # PLOTS
    # ==========================================================

    @staticmethod
    def plot_errors(epochs, errors, output_dir):
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, errors, label="Error")
        plt.grid()
        plt.savefig(os.path.join(output_dir, "error_plot.png"))
        plt.close()

    @staticmethod
    def plot_accuracy(epochs, accuracy, output_dir):
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, accuracy, label="Accuracy")
        plt.grid()
        plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
        plt.close()

    def save_model(self, path):

        with open(path, "w", encoding="utf-8") as f:

            f.write("MLP_MODEL\n\n")

            f.write(
                "LAYERS "
                + " ".join(map(str, self.layers))
                + "\n"
            )

            f.write(
                f"LEARNING_RATE "
                f"{self.learning_rate}\n"
            )

            f.write(
                f"MOMENTUM "
                f"{self.momentum}\n"
            )

            f.write(
                f"USE_BIAS "
                f"{int(self.use_bias)}\n"
            )

            f.write(
                f"USE_MOMENTUM "
                f"{int(self.use_momentum)}\n\n"
            )

            # ==========================================================
            # SAVE WEIGHTS PER LAYER / NEURON
            # ==========================================================

            for layer_index, weights in enumerate(self.weights):

                f.write("=" * 80 + "\n")
                f.write(f"LAYER {layer_index}\n")
                f.write("=" * 80 + "\n")

                input_size = self.layers[layer_index]

                if self.use_bias:
                    input_size += 1

                output_size = self.layers[layer_index + 1]

                for neuron_index in range(output_size):

                    f.write(f"\nNEURON {neuron_index}\n")

                    base = neuron_index * input_size

                    neuron_weights = weights[
                                     base: base + input_size
                                     ]

                    for weight_index, weight in enumerate(neuron_weights):

                        if (
                                self.use_bias
                                and weight_index == input_size - 1
                        ):
                            f.write(
                                f"BIAS_WEIGHT {weight}\n"
                            )
                        else:
                            f.write(
                                f"W{weight_index} {weight}\n"
                            )

                f.write("\n")

    @staticmethod
    def load_model(path):

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]

        layers = []

        learning_rate = 0.6
        momentum = 0.0

        use_bias = True
        use_momentum = False

        weight_blocks = []

        current_layer_weights = []

        for line in lines:

            if not line:
                continue

            if line.startswith("="):
                continue

            # ==========================================================
            # CONFIG
            # ==========================================================

            if line.startswith("LAYERS"):

                layers = list(
                    map(int, line.split()[1:])
                )

            elif line.startswith("LEARNING_RATE"):

                learning_rate = float(
                    line.split()[1]
                )

            elif line.startswith("MOMENTUM"):

                momentum = float(
                    line.split()[1]
                )

            elif line.startswith("USE_BIAS"):

                use_bias = bool(
                    int(line.split()[1])
                )

            elif line.startswith("USE_MOMENTUM"):

                use_momentum = bool(
                    int(line.split()[1])
                )

            # ==========================================================
            # NEW LAYER
            # ==========================================================

            elif line.startswith("LAYER"):

                if current_layer_weights:
                    weight_blocks.append(
                        current_layer_weights
                    )

                current_layer_weights = []

            # ==========================================================
            # IGNORE NEURON LABELS
            # ==========================================================

            elif line.startswith("NEURON"):

                continue

            # ==========================================================
            # WEIGHTS
            # ==========================================================

            elif (
                    line.startswith("W")
                    or line.startswith("BIAS_WEIGHT")
            ):

                parts = line.split()

                value = float(parts[1])

                current_layer_weights.append(value)

        # last layer
        if current_layer_weights:
            weight_blocks.append(
                current_layer_weights
            )

        # ==============================================================
        # CREATE NETWORK
        # ==============================================================

        net = MLP(
            layers=layers,
            learning_rate=learning_rate,
            momentum=momentum,
            use_bias=use_bias,
            use_momentum=use_momentum
        )

        net.weights = [
            np.array(w, dtype=float)
            for w in weight_blocks
        ]

        return net
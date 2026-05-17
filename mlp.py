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

        self.layers = layers

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.use_bias = use_bias
        self.use_momentum = use_momentum

        self.weights = []
        self.previous_updates = []

        self.outputs = []
        self.deltas = []

        self._initialize_weights(
            weight_init_min,
            weight_init_max
        )
        self.epoch_target = 0

    def _initialize_weights(self, wmin, wmax):
        for layer_index in range(len(self.layers) - 1):
            input_size = self.layers[layer_index]
            output_size = self.layers[layer_index + 1]

            if self.use_bias:
                input_size += 1

            count = input_size * output_size

            w = np.array([
                random.uniform(wmin, wmax)
                for _ in range(count)
            ], dtype=float)

            prev = np.zeros(count, dtype=float)

            self.weights.append(w)
            self.previous_updates.append(prev)

    def forward(self, inputs):
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
                s = np.dot(
                    weight_vector[
                        base: base + effective_input_size
                    ],
                    current_with_bias
                )
                out = sigmoid(s)
                next_outputs.append(out)

            self.outputs.append(next_outputs)

            current = next_outputs

        return current

    def backward(self, targets):

        self.deltas = [None] * len(self.layers)

        output_layer = len(self.layers) - 1

        output_deltas = []

        outputs = self.outputs[-1]

        for o, t in zip(outputs, targets):

            error = t - o

            delta = (
                error
                * sigmoid_derivative_from_output(o)
            )

            output_deltas.append(delta)

        self.deltas[output_layer] = output_deltas

        # ==============================================================
        # HIDDEN LAYERS
        # ==============================================================

        for layer in range(output_layer - 1, 0, -1):

            current_deltas = []

            current_outputs = self.outputs[layer]

            next_deltas = self.deltas[layer + 1]

            weights = self.weights[layer]

            current_size = self.layers[layer]
            next_size = self.layers[layer + 1]

            effective_current_size = (
                current_size + 1
                if self.use_bias
                else current_size
            )

            for neuron in range(current_size):

                error_sum = 0.0

                for next_neuron in range(next_size):

                    idx = (
                        next_neuron
                        * effective_current_size
                        + neuron
                    )

                    error_sum += (
                        weights[idx]
                        * next_deltas[next_neuron]
                    )

                delta = (
                    error_sum
                    * sigmoid_derivative_from_output(
                        current_outputs[neuron]
                    )
                )

                current_deltas.append(delta)

            self.deltas[layer] = current_deltas

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
                    grad = (
                        self.learning_rate
                        * deltas[neuron]
                        * inputs[i]
                    )

                    idx = base + i

                    if self.use_momentum:
                        update = (
                            grad
                            + self.momentum
                            * self.previous_updates[
                                layer_index
                            ][idx]
                        )

                        self.weights[layer_index][idx] += update

                        self.previous_updates[
                            layer_index
                        ][idx] = update

                    else:
                        self.weights[layer_index][idx] += grad

    def evaluate(self, dataset):

        predictions = []

        total_error = 0.0

        for x, y in zip(dataset.X, dataset.Y):

            out = self.forward(x)

            predictions.append(out)

            total_error += mse(y, out)

        accuracy = Metrics.accuracy(
            dataset.Y,
            predictions
        )

        return accuracy, total_error

    def train(
            self,
            dataset,
            epochs=5000,
            target_error=0.001,
            shuffle=True,
            stop_on_target=True,
            patience=500,
            log_every=1,
            output_dir="outputs"
    ):

        ensure_dir(output_dir)

        training_csv = os.path.join(
            output_dir,
            "training_log.csv"
        )

        history_epochs = []
        history_errors = []
        history_accuracy = []

        best_error = float("inf")
        epochs_without_improvement = 0

        with open(training_csv, "w", newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow([
                "epoch",
                "global_error",
                "accuracy"
            ])

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

                accuracy, evaluated_error = (
                    self.evaluate(dataset)
                )

                history_epochs.append(epoch)
                history_errors.append(evaluated_error)
                history_accuracy.append(accuracy)

                if epoch % log_every == 0:

                    writer.writerow([
                        epoch,
                        evaluated_error,
                        accuracy
                    ])

                    print(
                        f"[TRAIN] "
                        f"epoch={epoch} "
                        f"error={evaluated_error:.6f} "
                        f"accuracy={accuracy:.4f}"
                    )

                if evaluated_error < best_error:

                    best_error = evaluated_error
                    epochs_without_improvement = 0

                else:

                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:

                    print(
                        "\nEarly stopping triggered."
                    )

                    break

                if (
                        stop_on_target
                        and evaluated_error <= target_error
                ):

                    print(
                        f"\nTarget error reached "
                        f"at epoch {epoch}"
                    )
                    self.epoch_target = epoch

                    break

        self.epoch_target = epoch

        self.plot_errors(
            history_epochs,
            history_errors,
            output_dir
        )

        self.plot_accuracy(
            history_epochs,
            history_accuracy,
            output_dir
        )

    def test(
            self,
            dataset,
            output_dir="outputs"
    ):

        ensure_dir(output_dir)

        predictions = []

        test_dump = os.path.join(
            output_dir,
            "test_results.txt"
        )

        with open(test_dump, "w", encoding="utf-8") as f:

            for i in range(len(dataset.X)):

                x = dataset.X[i]
                y = dataset.Y[i]

                out = self.forward(x)

                predictions.append(out)

                err = mse(y, out)

                f.write("=" * 80 + "\n")

                f.write(f"SAMPLE {i}\n\n")

                f.write(f"INPUT:\n{x}\n\n")

                f.write(f"TARGET:\n{y}\n\n")

                f.write(f"OUTPUT:\n{out}\n\n")

                f.write(f"PATTERN ERROR:\n{err}\n\n")

                f.write("LAYER OUTPUTS:\n")

                for layer_index, layer_out in enumerate(self.outputs):

                    f.write(
                        f"Layer {layer_index}: "
                        f"{layer_out}\n"
                    )

                f.write("\nWEIGHTS:\n")

                for layer_index, weights in enumerate(self.weights):

                    f.write(
                        f"Layer {layer_index}:\n"
                    )

                    f.write(
                        f"{weights.tolist()}\n\n"
                    )

        num_classes = len(dataset.Y[0])

        cm = Metrics.confusion_matrix(
            dataset.Y,
            predictions,
            num_classes
        )

        precision, recall, f1 = (
            Metrics.precision_recall_f1(cm)
        )

        accuracy = Metrics.accuracy(
            dataset.Y,
            predictions
        )

        self.save_confusion_matrix(
            cm,
            output_dir
        )

        report_file = os.path.join(
            output_dir,
            "report.txt"
        )

        with open(report_file, "w") as f:

            f.write("=" * 80 + "\n")
            f.write("MLP TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"Architecture: {self.layers}\n"
            )

            f.write(
                f"Learning rate: "
                f"{self.learning_rate}\n"
            )

            f.write(
                f"Momentum: "
                f"{self.momentum}\n"
            )

            f.write(
                f"Bias: "
                f"{self.use_bias}\n"
            )

            f.write(
                f"Accuracy: "
                f"{accuracy}\n"
            )

            f.write(
                f"Target error reached at epoch: "
                f"{self.epoch_target}\n\n"
            )

            f.write("CONFUSION MATRIX\n")

            for row in cm:
                f.write(f"{row}\n")

            f.write("\n")

            for i in range(num_classes):

                f.write(f"CLASS {i}\n")

                f.write(
                    f"Precision: "
                    f"{precision[i]}\n"
                )

                f.write(
                    f"Recall: "
                    f"{recall[i]}\n"
                )

                f.write(
                    f"F1: "
                    f"{f1[i]}\n\n"
                )

        print("\nTEST FINISHED")
        print(f"Accuracy: {accuracy:.4f}")

    def save_confusion_matrix(
            self,
            matrix,
            output_dir
    ):

        path = os.path.join(
            output_dir,
            "confusion_matrix.csv"
        )

        with open(path, "w", newline="") as f:

            writer = csv.writer(f)

            for row in matrix:
                writer.writerow(row)

    def plot_errors(
            self,
            epochs,
            errors,
            output_dir
    ):

        plt.figure(figsize=(12, 7))

        plt.plot(
            epochs,
            errors,
            color="blue",
            linewidth=2,
            label="Global Error"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Error")

        plt.title("Training Error")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "error_plot.png"
            ),
            dpi=300
        )

        plt.close()

    def plot_accuracy(
            self,
            epochs,
            accuracy,
            output_dir
    ):

        plt.figure(figsize=(12, 7))

        plt.plot(
            epochs,
            accuracy,
            color="green",
            linewidth=2
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.title("Training Accuracy")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "accuracy_plot.png"
            ),
            dpi=300
        )

        plt.close()

    def save_model(self, path):

        with open(path, "w", encoding="utf-8") as f:

            f.write("MLP_MODEL\n")

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
                f"{int(self.use_momentum)}\n"
            )

            for i, weights in enumerate(self.weights):

                f.write(f"LAYER {i}\n")

                f.write(
                    " ".join(
                        map(
                            str,
                            weights.tolist()
                        )
                    ) + "\n"
                )

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

        current_layer = None

        for line in lines:

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

            elif line.startswith("LAYER"):

                current_layer = []

                weight_blocks.append(current_layer)

            else:

                if current_layer is not None:

                    current_layer.extend(
                        map(float, line.split())
                    )

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
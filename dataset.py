import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    X: list
    Y: list
    labels: list | None = None

<<<<<<< HEAD

class DatasetLoader:

    @staticmethod
    def load_txt_dataset(path):

=======
class DatasetLoader:
    @staticmethod
    def load_txt_dataset(path):
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
        X = []
        Y = []

        class_map = {}
        class_counter = 0

<<<<<<< HEAD
        with open(path, "r", encoding="utf-8") as f:
=======
        with open(path, "r") as f:
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086

            for line in f:

                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]

<<<<<<< HEAD
=======
                # ==========================================================
                # IRIS STYLE
                # ==========================================================

>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
                if len(parts) >= 2 and "->" not in line:

                    features = list(map(float, parts[:-1]))

                    class_name = parts[-1]

                    if class_name not in class_map:
                        class_map[class_name] = class_counter
                        class_counter += 1

                    X.append(features)
                    Y.append(class_name)

<<<<<<< HEAD
=======
                # ==========================================================
                # GENERIC STYLE
                # ==========================================================

>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
                else:

                    split = line.split("->")

                    if len(split) != 2:
                        raise ValueError(f"Invalid line: {line}")

                    left = split[0].strip()
                    right = split[1].strip()

<<<<<<< HEAD
                    x = list(map(float, map(str.strip, left.split(","))))
                    y = list(map(float, map(str.strip, right.split(","))))
=======
                    x = [v.strip() for v in left.split(",")]
                    y = [v.strip() for v in right.split(",")]
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086

                    X.append(x)
                    Y.append(y)

<<<<<<< HEAD
=======
        # ==============================================================
        # ONE HOT
        # ==============================================================
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
        if len(Y) > 0 and isinstance(Y[0], str):

            num_classes = len(class_map)

            Y_encoded = []

            for label in Y:

                vec = [0.0] * num_classes
<<<<<<< HEAD
=======

>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
                vec[class_map[label]] = 1.0

                Y_encoded.append(vec)

            Y = Y_encoded

        return Dataset(X, Y)

    @staticmethod
<<<<<<< HEAD
=======
    def normalize_minmax(dataset: Dataset):

        X = np.array(dataset.X, dtype=float)

        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        denom = maxs - mins
        denom[denom == 0.0] = 1.0

        Xn = (X - mins) / denom

        dataset.X = Xn.tolist()

        return mins, maxs

    @staticmethod
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
    def stratified_split(dataset, train_ratio=0.8, seed=42):

        random.seed(seed)

        class_groups = {}

        for x, y in zip(dataset.X, dataset.Y):

            cls = np.argmax(y)

            if cls not in class_groups:
                class_groups[cls] = []

            class_groups[cls].append((x, y))

<<<<<<< HEAD
        train_X, train_Y = [], []
        test_X, test_Y = [], []
=======
        train_X = []
        train_Y = []

        test_X = []
        test_Y = []
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086

        for cls in class_groups:

            samples = class_groups[cls]

            random.shuffle(samples)

            split_index = int(len(samples) * train_ratio)

            train = samples[:split_index]
            test = samples[split_index:]

            for x, y in train:
                train_X.append(x)
                train_Y.append(y)

            for x, y in test:
                test_X.append(x)
                test_Y.append(y)

<<<<<<< HEAD
        return Dataset(train_X, train_Y), Dataset(test_X, test_Y)
=======
        return (
            Dataset(train_X, train_Y),
            Dataset(test_X, test_Y)
        )
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086

    @staticmethod
    def save_dataset(dataset, path):

        with open(path, "w", encoding="utf-8") as f:

            for x, y in zip(dataset.X, dataset.Y):

<<<<<<< HEAD
                left = ",".join(map(str, x))
                right = ",".join(map(str, y))

                f.write(f"{left} -> {right}\n")

    @staticmethod
    def fit_minmax(dataset):

        X = np.array(dataset.X, dtype=float)

        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        return mins, maxs

    @staticmethod
    def apply_minmax(dataset, mins, maxs):

        X = np.array(dataset.X, dtype=float)

        denom = maxs - mins
        denom = np.where(denom == 0.0, 1.0, denom)

        Xn = (X - mins) / denom

        dataset.X = Xn.tolist()
=======
                f.write(
                    f"{x} -> {y}\n"
                )

>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086

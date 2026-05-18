class Metrics:

    @staticmethod
    def argmax(vec):
        return max(range(len(vec)), key=lambda i: vec[i])

    @staticmethod
    def accuracy(y_true, y_pred):

<<<<<<< HEAD
        if len(y_true) == 0:
            return 0.0

=======
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
        correct = 0

        for t, p in zip(y_true, y_pred):

            if Metrics.argmax(t) == Metrics.argmax(p):
                correct += 1

        return correct / len(y_true)

    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes):

        matrix = [
            [0 for _ in range(num_classes)]
            for _ in range(num_classes)
        ]

        for t, p in zip(y_true, y_pred):

            ti = Metrics.argmax(t)
            pi = Metrics.argmax(p)

            matrix[ti][pi] += 1

        return matrix

    @staticmethod
    def precision_recall_f1(conf_matrix):

        num_classes = len(conf_matrix)

        precision = []
        recall = []
        f1 = []

        for c in range(num_classes):

            tp = conf_matrix[c][c]

            fp = sum(conf_matrix[r][c]
                     for r in range(num_classes)) - tp

            fn = sum(conf_matrix[c][r]
                     for r in range(num_classes)) - tp

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f = (
                2 * p * r / (p + r)
                if (p + r) > 0
                else 0.0
            )

            precision.append(p)
            recall.append(r)
            f1.append(f)

<<<<<<< HEAD
        return precision, recall, f1

    @staticmethod
    def per_class_accuracy(conf_matrix):

        results = []

        for i, row in enumerate(conf_matrix):

            correct = row[i]
            total = sum(row)

            acc = correct / total if total > 0 else 0.0

            results.append(acc)

        return results

    @staticmethod
    def save_confusion_matrix(matrix, output_dir, filename="confusion_matrix.csv"):

        import csv
        import os

        path = os.path.join(output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            for row in matrix:
                writer.writerow(row)
=======
        return precision, recall, f1
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086

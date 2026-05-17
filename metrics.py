class Metrics:

    @staticmethod
    def argmax(vec):
        return max(range(len(vec)), key=lambda i: vec[i])

    @staticmethod
    def accuracy(y_true, y_pred):

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

        return precision, recall, f1
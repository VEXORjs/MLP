import os

from utilities import ensure_dir
from mlp import MLP
from dataset import Dataset


def run_autoencoder_experiment(
        output_dir="outputs_autoencoder"
):
    ensure_dir(output_dir)

    patterns = Dataset(
        X=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        Y=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    experiments = [
        (0.9, 0.0),
        (0.6, 0.0),
        (0.2, 0.0),
        (0.9, 0.6),
        (0.2, 0.9),
    ]

    for lr, mom in experiments:

        name = f"lr_{lr}_mom_{mom}"

        exp_dir = os.path.join(
            output_dir,
            name
        )

        ensure_dir(exp_dir)

        net = MLP(
            layers=[4, 2, 4],
            learning_rate=lr,
            momentum=mom,
            use_bias=True,
            use_momentum=(mom > 0.0)
        )

        net.train(
            patterns,
            epochs=10000,
            target_error=0.001,
            shuffle=True,
            stop_on_target=True,
            output_dir=exp_dir
        )

        net.test(
            patterns,
            output_dir=exp_dir
        )

        net.save_model(
            os.path.join(
                exp_dir,
                "model.txt"
            )
        )


import os

from utilities import ensure_dir
from mlp import MLP
from dataset import Dataset

def run_autoencoder_experiment(output_dir="outputs_autoencoder"):

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
        (0.9, 0.0, True),
        (0.6, 0.0, True),
        (0.2, 0.0, True),
        (0.9, 0.6, True),
        (0.2, 0.9, True),

        # --- KEY COMPARISON ---
        (0.6, 0.0, False),
    ]

    for lr, mom, bias in experiments:

        name = f"lr_{lr}_mom_{mom}_bias_{bias}"

        exp_dir = os.path.join(output_dir, name)
        ensure_dir(exp_dir)

        net = MLP(
            layers=[4, 2, 4],
            learning_rate=lr,
            momentum=mom,
            use_bias=bias,
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

        net.test(patterns, output_dir=exp_dir)

        # --- HIDDEN LAYER ANALYSIS ---
        hidden_file = os.path.join(exp_dir, "hidden_outputs.txt")

        with open(hidden_file, "w") as f:

            for i, x in enumerate(patterns.X):

                net.forward(x)
                hidden = net.outputs[1]

                f.write(f"Input {i}: {x}\n")
                f.write(f"Hidden layer: {hidden}\n\n")

        net.save_model(os.path.join(exp_dir, "model.txt"))
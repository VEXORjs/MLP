import os

from utilities import ensure_dir
from mlp import MLP
from dataset import Dataset

<<<<<<< HEAD
def run_autoencoder_experiment(output_dir="outputs_autoencoder"):

=======

def run_autoencoder_experiment(
        output_dir="outputs_autoencoder"
):
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
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
<<<<<<< HEAD
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
=======
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

>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
        ensure_dir(exp_dir)

        net = MLP(
            layers=[4, 2, 4],
            learning_rate=lr,
            momentum=mom,
<<<<<<< HEAD
            use_bias=bias,
=======
            use_bias=True,
>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086
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

<<<<<<< HEAD
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
=======
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

>>>>>>> 8eb92a82eb1b591f0a4c472e860caf397df2b086

import argparse
import os

import autoencoder
from mlp import MLP
from utilities import ensure_dir
from dataset import DatasetLoader

# =============================================================================
# MENU HELPERS
# =============================================================================

def build_architecture_from_user():
    print("\n=== NETWORK ARCHITECTURE ===")

    input_size = int(input(
        "Input neurons "
        "(Iris = 4): "
    ))

    hidden_layers_count = int(input(
        "Hidden layers "
        "(recommended 1-3): "
    ))

    while hidden_layers_count < 0:

        hidden_layers_count = int(input(
            "Hidden layers must be >= 0: "
        ))

    architecture = [input_size]

    for i in range(hidden_layers_count):

        neurons = int(input(
            f"Neurons in hidden layer {i+1} "
            f"(recommended 4-32): "
        ))

        while neurons <= 0:

            neurons = int(input(
                "Neuron count must be > 0: "
            ))

        architecture.append(neurons)

    output_size = int(input(
        "Output neurons "
        "(Iris = 3): "
    ))

    architecture.append(output_size)

    print(
        f"\nFinal architecture: "
        f"{architecture}"
    )

    return architecture


# =============================================================================
# MENU
# =============================================================================

def interactive_menu():

    while True:

        print("\n" + "=" * 80)
        print("MLP FRAMEWORK")
        print("=" * 80)

        print("1. Train network")
        print("2. Test network")
        print("3. Autoencoder study")
        print("4. Exit")

        choice = input("\nChoice: ").strip()

        # =====================================================================
        # TRAIN
        # =====================================================================
        if choice == "1":

            train_file = input(
                "\nDataset path: "
            ).strip()

            output_dir = input(
                "Output directory: "
            ).strip()

            ensure_dir(output_dir)

            dataset = DatasetLoader.load_txt_dataset(
                train_file
            )

            DatasetLoader.normalize_minmax(dataset)

            train_dataset, test_dataset = (
                DatasetLoader.stratified_split(
                    dataset,
                    train_ratio=0.8
                )
            )

            DatasetLoader.save_dataset(
                train_dataset,
                os.path.join(
                    output_dir,
                    "train_split.txt"
                )
            )

            DatasetLoader.save_dataset(
                test_dataset,
                os.path.join(
                    output_dir,
                    "test_split.txt"
                )
            )

            layers = build_architecture_from_user()

            lr = float(input(
                "\nLearning rate "
                "(recommended 0.1-0.6)\n"
                "Lower = slower but more stable\n"
                "Higher = faster but may diverge\n"
                "Value: "
            ))

            momentum = float(input(
                "\nMomentum "
                "(recommended 0.0-0.9)\n"
                "Higher may accelerate learning\n"
                "Too high may destabilize training\n"
                "Value: "
            ))

            epochs = int(input(
                "\nEpochs "
                "(recommended 1000-10000)\n"
                "Higher = potentially better accuracy\n"
                "but slower training\n"
                "Value: "
            ))

            target_error = float(input(
                "\nTarget error "
                "(recommended 0.001): "
            ))

            bias = (
                input(
                    "\nUse bias? (y/n): "
                )
                .lower()
                .startswith("y")
            )

            use_momentum = (
                input(
                    "Use momentum? (y/n): "
                )
                .lower()
                .startswith("y")
            )

            shuffle = (
                input(
                    "Shuffle every epoch? (y/n): "
                )
                .lower()
                .startswith("y")
            )

            net = MLP(
                layers=layers,
                learning_rate=lr,
                momentum=momentum,
                use_bias=bias,
                use_momentum=use_momentum
            )

            net.train(
                train_dataset,
                epochs=epochs,
                target_error=target_error,
                shuffle=shuffle,
                output_dir=output_dir
            )

            print("\nTesting on test dataset...\n")

            net.test(
                test_dataset,
                output_dir=output_dir
            )
            save_model = (
                input(
                    "\nSave model? (y/n): "
                )
                .lower()
                .startswith("y")
            )
            if save_model:

                model_path = input(
                    "Model save path "
                    "(example: models/model.txt): "
                ).strip()

                ensure_dir(
                    os.path.dirname(model_path)
                    if os.path.dirname(model_path)
                    else "."
                )

                net.save_model(model_path)

                print(
                    f"Model saved to: "
                    f"{model_path}"
                )

        # =====================================================================
        # TEST SECTION
        # =====================================================================
        elif choice == "2":
            model_path = input(
                "\nModel path: "
            ).strip()

            test_file = input(
                "Test dataset path: "
            ).strip()

            output_dir = input(
                "Output directory: "
            ).strip()

            dataset = DatasetLoader.load_txt_dataset(
                test_file
            )

            DatasetLoader.normalize_minmax(dataset)

            net = MLP.load_model(model_path)

            net.test(
                dataset,
                output_dir=output_dir
            )

        # =====================================================================
        # AUTOENCODER
        # =====================================================================
        elif choice == "3":

            output_dir = input(
                "\nOutput directory: "
            ).strip()

            autoencoder.run_autoencoder_experiment(
                output_dir
            )

        # =====================================================================
        # EXIT
        # =====================================================================
        elif choice == "4":
            break
        else:
            print("\nInvalid choice.")


# =============================================================================
# CLI
# =============================================================================
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--menu",
        action="store_true"
    )

    args = parser.parse_args()

    interactive_menu()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    cli()
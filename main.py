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
        print("1. Prepare & Split dataset (Train/Test split)")
        print("2. Train network (Using training dataset)")
        print("3. Test network (Choose model & test dataset)")
        print("4. Autoencoder study")
        print("5. Exit")

        choice = input("\nChoice: ").strip()

        # =====================================================================
        # 1. PREPARE & SPLIT DATASET
        # =====================================================================
        if choice == "1":
            print("\nAvailable datasets in current directory:")

            raw_file = input("\nDataset path to split: ").strip()
            output_dir = input("Output directory for splits: ").strip()
            ensure_dir(output_dir)

            # Ładowanie i normalizacja
            dataset = DatasetLoader.load_txt_dataset(raw_file)
            DatasetLoader.normalize_minmax(dataset)

            # Podział
            train_dataset, test_dataset = DatasetLoader.stratified_split(
                dataset, train_ratio=0.8
            )

            # Zapis osobnych plików na dysk
            train_path = os.path.join(output_dir, "train_split.txt")
            test_path = os.path.join(output_dir, "test_split.txt")

            DatasetLoader.save_dataset(train_dataset, train_path)
            DatasetLoader.save_dataset(test_dataset, test_path)

            print(f"\n[SUCCESS] Datasets split successfully!")
            print(f"-> Train split saved to: {train_path}")
            print(f"-> Test split saved to: {test_path}")

        # =====================================================================
        # 2. TRAIN NETWORK
        # =====================================================================
        elif choice == "2":
            print("\nRemember to provide the path to your split file (e.g., output/train_split.txt)")
            train_file = input("\nProvide TRAIN dataset path (e.g., output/train_split.txt): ").strip()
            output_dir = input("Output directory for logs: ").strip()
            ensure_dir(output_dir)

            # Ładowanie wyłącznie zbioru treningowego
            train_dataset = DatasetLoader.load_txt_dataset(train_file)

            # Konfiguracja sieci
            layers = build_architecture_from_user()

            lr = float(input("\nLearning rate (recommended 0.1-0.6): "))
            momentum = float(input("\nMomentum (recommended 0.0-0.9): "))
            epochs = int(input("\nEpochs (recommended 1000-10000): "))
            target_error = float(input("\nTarget error (recommended 0.001): "))
            bias = input("\nUse bias? (y/n): ").lower().startswith("y")
            use_momentum = input("Use momentum? (y/n): ").lower().startswith("y")
            shuffle = input("Shuffle every epoch? (y/n): ").lower().startswith("y")

            net = MLP(
                layers=layers,
                learning_rate=lr,
                momentum=momentum,
                use_bias=bias,
                use_momentum=use_momentum
            )

            # Trenowanie na zbiorze treningowym
            print("\nStarting training...")
            net.train(
                train_dataset,
                epochs=epochs,
                target_error=target_error,
                shuffle=shuffle,
                output_dir=output_dir
            )

            # Opcjonalne zapisanie modelu po treningu
            save_model = input("\nSave trained model? (y/n): ").lower().startswith("y")
            if save_model:
                model_path = input("Model save path (example: models/my_model.txt): ").strip()
                ensure_dir(os.path.dirname(model_path) if os.path.dirname(model_path) else ".")

                net.save_model(model_path)
                absulute_path = os.path.abspath(model_path) if model_path else None

                print(f"-> Trained model saved to relative path: {model_path}")
                print(f"[SUCCESS] Trained model saved to absolute path: {absulute_path}")

        # =====================================================================
        # 3. TEST NETWORK (Wybór modelu i zestawu testowego)
        # =====================================================================
        elif choice == "3":
            print("\nAvailable saved models in current directory:")

            model_path = input("\nPath to saved model file (e.g., models/my_model.txt): ").strip()
            test_file = input("Path to TEST dataset (e.g., output/test_split.txt): ").strip()
            output_dir = input("Output directory for test results: ").strip()
            ensure_dir(output_dir)

            # 1. Wczytanie zbioru testowego
            test_dataset = DatasetLoader.load_txt_dataset(test_file)

            # 2. Odtworzenie modelu z pliku
            try:
                print("\nLoading model...")
                net = MLP.load_model(model_path)

                print(f"Testing model '{model_path}' on dataset '{test_file}'...\n")
                net.test(
                    test_dataset,
                    output_dir=output_dir
                )
                print("[SUCCESS] Testing completed. Results saved in output directory.")
            except Exception as e:
                print(f"[ERROR] Could not load or test the model: {e}")

        # =====================================================================
        # 4. AUTOENCODER STUDY
        # =====================================================================
        elif choice == "4":

            output_dir = input(
                "\nOutput directory: "
            ).strip()

            autoencoder.run_autoencoder_experiment(
                output_dir
            )

        # =====================================================================
        # 5. EXIT
        # =====================================================================
        elif choice == "5":
            print("\nExiting MLP Framework. Goodbye!")
            break

        else:
            print("\n[WARNING] Invalid choice! Please select a number between 1 and 5.")
# =============================================================================
# CLI
# =============================================================================
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--menu",
        action="store_true"
    )

    parser.parse_args()

    interactive_menu()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    cli()
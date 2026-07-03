from src.chembl_downloader import ChEMBLDownloader
from src.preprocess import preprocess
from src.descriptors import calculate_descriptors
from src.feature_selection import feature_selection
from src.train import train_model
from src.evaluate import evaluate_model


def main():

    print("\n======================================")
    print("          AutoQSAR Pipeline")
    print("======================================")

    # Step 1 - Download
    downloader = ChEMBLDownloader()
    downloader.run()

    # Step 2 - Clean Dataset
    preprocess()

    # Step 3 - Calculate Descriptors
    calculate_descriptors()

    # Step 4 - Feature Selection
    feature_selection()

    # Step 5 - Train Model
    train_model()

    # Step 6 - Evaluate Model
    evaluate_model()

    print("\n======================================")
    print("     AutoQSAR Pipeline Complete!")
    print("======================================")


if __name__ == "__main__":
    main()
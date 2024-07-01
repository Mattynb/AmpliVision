"""Main function to generate data from the results folder and save it to a csv file."""

from src.utils import Utils
from src.data_extractor import DataExtractor
from src.data_generator import DataGenerator


def main(sample_type: str, results_path: str, save_path: str) -> None:
    """Main function to generate data from the results folder and save it to a csv file."""

    # Extract data from the results folder, getting the fingerprints
    data_extractor = DataExtractor(sample_type, extract_from=results_path)
    combined_fingerprints = data_extractor.extract(display=1)

    # Use the extracted fingerprints to generate realistic data
    data_generator = DataGenerator(combined_fingerprints)
    corr_pts = data_generator.generate_n_points(20)

    # Get the original RGB values
    pts = Utils.subtract_255(corr_pts)

    # Display the generated points
    #Utils.visualize_generated_pts(pts, f"Generated {sample_type} PTS")
    Utils.plot_3d_generated_pts(pts, f"Generated {sample_type} PTS")

    # Save the generated points to a csv file
    # Utils.write_to_csv(sample_type, save_path,  pts)


if __name__ == '__main__':
    # Example usage
    SAMPLE_TYPES = [
        "breast", "control", "lung",
        "ovarian", "prostate", "skin", "thyroid"
    ]

    SAVE_PATH = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/generated_results"
    RESULTS_FOLDER_PATH = r"""
    C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/results/06-27-2024/""".strip()

    for SAMPLE_TYPE in SAMPLE_TYPES:
        print(f"generating {SAMPLE_TYPE} results")
        main(SAMPLE_TYPE, RESULTS_FOLDER_PATH, SAVE_PATH)

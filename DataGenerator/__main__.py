"""Main function to generate data from the results folder and save it to a csv file."""

from src.utils import Utils
from src.data_extractor import DataExtractor
from src.data_generator import DataGenerator


def main(
    n_pts: int,
    sample_type: str,
    results_path: str,
    class_n: int,
    display: bool = False
) -> None:
    """Main function to generate data from the results folder and save it to a csv file."""

    # Extract data from the results folder, getting the fingerprints
    data_extractor = DataExtractor(sample_type, extract_from=results_path)
    combined_fingerprints = data_extractor.extract(display=0)

    # Use the extracted fingerprints to generate realistic data
    data_generator = DataGenerator(combined_fingerprints)
    corr_pts = data_generator.generate_points(n_pts)

    # Get the original RGB values
    pts = Utils.subtract_255(corr_pts)

    if display:
        Utils.display_fingerprint(combined_fingerprints)
        Utils.visualize_fingerprints_with_colors(combined_fingerprints)
        Utils.visualize_generated_pts(pts, f"Generated {sample_type} PTS")
        Utils.plot_3d_generated_pts(pts, f"Generated {sample_type} PTS")

    # return the points as flat json to be saved in csv
    return Utils.comb_pts_as_flat_json(corr_pts, class_n)


if __name__ == '__main__':
    # Constants
    N_PTS = 1000
    DISPLAY = False
    SAVE_PATH = r"data/generated_results"
    RESULTS_FOLDER_PATH = r"data/results/06-27-2024/"

    SAMPLE_TYPES = [
        "breast", "control", "lung",
        "ovarian", "prostate", "skin", "thyroid"
    ]

    print(f"Extracting data from {RESULTS_FOLDER_PATH}", end='\n\n')
    print(f"Generating {N_PTS} pts for each type\ntypes are: ", end='')
    print(*SAMPLE_TYPES, sep=', ', end='\n\n')

    # Generate data for each sample type
    flat_data = []
    for i, SAMPLE_TYPE in enumerate(SAMPLE_TYPES):
        flat_data.append(
            main(N_PTS, SAMPLE_TYPE, RESULTS_FOLDER_PATH, i, DISPLAY)
        )

    # Save the generated flat_json data to a csv file
    Utils.write_to_csv(SAVE_PATH, data=flat_data)
    print(f"Saving to {SAVE_PATH}")

"""Main function to generate data from the results folder and save it to a csv file."""

from re import S
import sys

from matplotlib.dates import SA
from src.utils import Utils
from src.data_extractor import DataExtractor
from src.data_generator import DataGenerator

def main(
    n_pts: int,
    sample_type: str,
    results_path: str,
    class_n: int,
    last_run: bool
) -> None:
    """Main function to generate data from the results folder and save it to a csv file."""

    # Extract data from the results folder, getting the fingerprints
    data_extractor = DataExtractor(sample_type, extract_from=results_path)
    combined_fingerprints = data_extractor.extract(display=0)

    # Use the extracted fingerprints to generate realistic data
    data_generator = DataGenerator(combined_fingerprints)
    corr_pts = data_generator.generate_n_points(n_pts)

    #Utils.print_generated_pts(corr_pts)

    # Get the original RGB values
    pts = Utils.subtract_255(corr_pts)

    # Display the generated points
    #Utils.visualize_generated_pts(pts, f"Generated {sample_type} PTS")
    #Utils.plot_3d_generated_pts(pts, f"Generated {sample_type} PTS")  
    
    
    return Utils.comb_pts_as_flat_json(corr_pts, class_n)

if __name__ == '__main__':
    # Constants
    SAVE_PATH = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/generated_results"
    
    RESULTS_FOLDER_PATH = r"""
    C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/results/06-27-2024/
    """.strip()

    SAMPLE_TYPES = [
        "breast", "control", "lung",
        "ovarian", "prostate", "skin", "thyroid"
    ]

    # Get vars from stdin
    N_PTS = int(sys.argv[sys.argv.index("-n") + 1])
    SAVE_PATH = sys.argv[sys.argv.index("-s") + 1]
    RESULTS_FOLDER_PATH = sys.argv[sys.argv.index("-r") + 1]

    print(f"Generating {N_PTS} points for each sample type")
    #print(f"Saving to {SAVE_PATH}")
    #print(f"Extracting data from {RESULTS_FOLDER_PATH}")
   
    # Generate data for each sample type
    flat_data = []
    for i, SAMPLE_TYPE in enumerate(SAMPLE_TYPES):
        #print(f"generating {SAMPLE_TYPE} results")
        flat_data.append(
            main(N_PTS, SAMPLE_TYPE, RESULTS_FOLDER_PATH, i, i == (len(SAMPLE_TYPES) - 1))
        )

    # Save the generated flat_json data to a csv file
    Utils.write_to_csv(SAVE_PATH, data=flat_data)

    
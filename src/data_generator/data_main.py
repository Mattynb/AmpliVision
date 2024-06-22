from _src.data_extractor import DataExtractor
from _src.data_generator import DataGenerator
from _src.utils.fingerprint_utils import visualize_fingerprints_with_colors
from _src.utils.visualize_generated import visualize_generated_pts, print_generated_pts
from _src.utils.utils_csv import write_to_csv

import matplotlib.pyplot as plt


if __name__ == '__main__':
    sample_type = "IMG"
    results_folder_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/results/06-20-2024/"
    save_generated_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/generated_results"

    data_extractor = DataExtractor(sample_type, extract_from=results_folder_path)
    combined_fingerprints = data_extractor.extract(display=0)


    data_generator = DataGenerator(combined_fingerprints)
    pts = data_generator.generate_n_points(10)
    visualize_fingerprints_with_colors(combined_fingerprints)
    #print_generated_pts(pts, subtract=255)

    write_to_csv(save_generated_path,  pts)


"""TODO: Understand why brg is used instead of rgb in fingerprint, might be phase B problem"""
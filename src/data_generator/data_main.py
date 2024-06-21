from _src.data_extractor import DataExtractor
from _src.data_generator import DataGenerator
from _src.utils.visualize_generated import visualize_generated_pts, print_generated_pts
from _src.utils.fingerprint_utils import display_fingerprint
from _src.utils.utils_csv import write_to_csv

if __name__ == '__main__':
    sample_type = "IMG"
    results_folder_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/results/06-20-2024/"
    save_generated_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/generated_results"

    data_extractor = DataExtractor(sample_type, extract_from=results_folder_path)
    combined_fingerprints = data_extractor.extract(display=1)
    

    data_generator = DataGenerator(combined_fingerprints)
    pts = data_generator.generate_n_points(10)

    print_generated_pts(pts)
    visualize_generated_pts(pts)

    write_to_csv(save_generated_path,  pts)
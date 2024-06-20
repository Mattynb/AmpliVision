from src.data_extractor import DataExtractor
from src.data_generator import DataGenerator
from src.utils.visualize_generated import visualize_generated_pts, print_generated_pts
from src.utils.fingerprint_utils import display_fingerprint
from src.data_generator.src.utils.utils_csv import write_to_csv

if __name__ == '__main__':
    sample_type = "DENV"
    results_folder_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/results/06-06-2024/"
    save_generated_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/generated_results"

    data_extractor = DataExtractor(sample_type, extract_from=results_folder_path)
    combined_fingerprints = data_extractor.extract_fingerprints_across_files()
    

    data_generator = DataGenerator(combined_fingerprints)
    pts = data_generator.generate_n_points(10)

    '''for type in pts:
        pts[type] = [[255-x for x in pair_list] for pair_list in pts[type]]'''
    print_generated_pts(pts)
    display_fingerprint(combined_fingerprints)
    visualize_generated_pts(pts)

    write_to_csv(save_generated_path,  pts)
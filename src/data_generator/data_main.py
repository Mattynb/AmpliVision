from _src.data_extractor import DataExtractor
from _src.data_generator import DataGenerator
from _src.utils.fingerprint_utils import visualize_fingerprints_with_colors
from _src.utils.visualize_generated import visualize_generated_pts, print_generated_pts, plot_3d_generated_pts
from _src.utils.utils_csv import write_to_csv

def subtract_255(rgb):
    new_rgb= {}
    for type in rgb:
        new_rgb[type] = []
        for row in rgb[type]:
            new_rgb[type].append([255 - x for x in row])
    return new_rgb

if __name__ == '__main__':
    sample_type = "IMG"
    results_folder_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/results/06-24-2024/"
    save_generated_path = r"C:/Users/Matheus/Desktop/NanoTechnologies_Lab/Phase A/data/generated_results"

    data_extractor = DataExtractor(sample_type, extract_from=results_folder_path)
    combined_fingerprints = data_extractor.extract(display=0)


    data_generator = DataGenerator(combined_fingerprints)
    corr_pts = data_generator.generate_n_points(1)
    pts = subtract_255(corr_pts)

    #visualize_fingerprints_with_colors(combined_fingerprints)
    #'''
    #print("\nPTS\n")
    #print_generated_pts(pts)
    #print("\nCORR PTS\n")
    #print_generated_pts(corr_pts)#'''
    
    #plot_3d_generated_pts(corr_pts, "Corr")
    plot_3d_generated_pts(pts, "PTS")
    #write_to_csv(save_generated_path,  pts)



    
"""TODO: Understand why brg is used instead of rgb in fingerprint, might be phase B problem"""
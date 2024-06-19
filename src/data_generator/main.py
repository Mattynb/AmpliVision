import dis
from src.data_extractor import DataExtractor
from src.data_generator import DataGenerator
from src.utils.fingerprint_utils import display_fingerprint

if __name__ == '__main__':
    sample_type = "DENV"
    results_folder_path = r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\results\06-06-2024\\"

    data_extractor = DataExtractor(sample_type, extract_from=results_folder_path)
    combined_fingerprints = data_extractor.extract_fingerprints_across_files()
    print(combined_fingerprints)

    data_generator = DataGenerator(combined_fingerprints)
    pts = data_generator.generate_n_points(10)

    print("\n","-"*10, "GENERATED DATA","-"*10,)
    for type in pts:
        print(f"{type}: ")
        for r in pts[type]:
            print(f"\t{r}")
    print("-"*30)
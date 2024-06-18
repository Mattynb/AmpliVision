import dis
from glob import glob
try:
    from src.data_generator.rgb_fingerprint_extractor import extract_fingerprint_from_single_csv, display_fingerprint
except ModuleNotFoundError:
    from rgb_fingerprint_extractor import extract_fingerprint_from_single_csv, display_fingerprint

#Workflow

# create class instance for each sample_type / disease type
    # extract and combine rgb fingerprint from each csv file for each sample_type

    # generate random data based on the rgb fingerprint's mean and standard deviation for each sample_type 

    # save the generated data to a csv file



class DataGenerator:
    def __init__(self, sample_type:str, results_folder_path:str):
        self.sample_type = sample_type
        self.results_folder_path = results_folder_path

    def load_csv_files(self, path:str = None)->list:
        """ load the csv files from the results folder"""
        if path is None:
            path = self.results_folder_path
        
        # find all the csv files in the folder that start with the sample_type
        csv_files = glob(path + f"{self.sample_type}*.csv")
        return csv_files

    def extract_fingerprints_across_files(self, path:str = None)->dict:
        """ extract the RGB fingerprint from the csv file"""

        # load the csv files
        csv_files = self.load_csv_files(path)
        
        # extract the RGB fingerprint from each csv file
        fingerprints=[]
        for csv_file in csv_files:
            fingerprints.append(extract_fingerprint_from_single_csv(csv_file))

        print("\n fingerprints: ", fingerprints)

        # append the RGB fingerprints
        appended_fingerprints = self.append_fingerprints(fingerprints)

        print("\n appended: ", appended_fingerprints)
        # combine the RGB fingerprints

        combined_fingerprints = self.combine_fingerprints(appended_fingerprints)
        print("\n combined: ", combined_fingerprints)

        display_fingerprint(combined_fingerprints)

    def append_fingerprints(self, fingerprints:list[dict[dict[list[int]]]])->dict[list[dict[list[int]]]]:
        """ groups the RGB fingerprints by block type
        
        Args:
            fingerprints (list): A list of RGB fingerprints
        
        fingerprints = 
        [
            {    
                'block_type1': 
                                {
                                    'r' : [mean_spot1, std_spot1, mean_spot2, std_spot2],
                                    'g' : [mean_spot1, std_spot1, mean_spot2, std_spot2],
                                    'b' : [mean_spot1, std_spot1, mean_spot2, std_spot2]
                                },
                ...more block_types
            }
        ]

        Returns:
            map: A dictionary containing the RGB values of each spot for each block type
            
        appended_fingerprints = {
                'block_type1': [ 
                                    { 
                                        'r' : [mean_spot1, std_spot1, mean_spot2, std_spot2], 
                                        'g' : [mean_spot1, std_spot1, mean_spot2, std_spot2], 
                                        'b' : [mean_spot1, std_spot1, mean_spot2, std_spot2] 
                                    },
                                    ...more image_results
                                ]
            ...more block_types
        }
        """

        combined_fingerprint = {}
        for fingerprint in fingerprints:
            for block_type in fingerprint:
                if block_type not in combined_fingerprint:
                    combined_fingerprint[block_type] = []

                combined_fingerprint[block_type].append(fingerprint[block_type])

        return combined_fingerprint
    
    def combine_fingerprints(self, appended_fingerprints:dict[list[dict[list[int]]]])->dict[dict[list[int]]]:
        """ combine the RGB fingerprints into a single RGB fingerprint for each block type
        
        appended_fingerprints = {
                'block_type1': [ 
                                    { 
                                        'r' : [mean_spot1, std_spot1, mean_spot2, std_spot2], 
                                        'g' : [mean_spot1, std_spot1, mean_spot2, std_spot2], 
                                        'b' : [mean_spot1, std_spot1, mean_spot2, std_spot2] 
                                    },
                                    ...more image_results
                                ]
            ...more block_types
        }
            
        
        """
     
        combined_fingerprint = {}
        for block_type in appended_fingerprints:
            
            # get the mean and standard deviation of the RGB values for each spot
            block_type_fingerprint = {}
            for rgb in ['r', 'g', 'b']:
                block_type_fingerprint[rgb] = self.get_mean_and_std(appended_fingerprints[block_type], rgb)

            combined_fingerprint[block_type] = block_type_fingerprint

        return combined_fingerprint

    def get_mean_and_std(self, image_results:list[dict[list[int]]], rgb:str)->list:
        """ calculate the mean and standard deviation of the RGB values for each spot
        
        Args:
            fingerprints (list): A list of RGB fingerprints
            rgb (str): The color channel
        
        
        image_results = [ 
                { 
                    'r' : [mean_spot1, std_spot1, mean_spot2, std_spot2], 
                    'g' : [mean_spot1, std_spot1, mean_spot2, std_spot2], 
                    'b' : [mean_spot1, std_spot1, mean_spot2, std_spot2] 
                },
                ...more image_results
            ]

        Returns:
            list: A list containing the mean and standard deviation of the RGB values for each spot


        """
        mean_and_std = []
        # calculate the mean and standard deviation of the RGB values for each spot
        for i in range(0, len(image_results[0][rgb]), 2):
            mean = 0
            std = 0
            for image_result in image_results:
                mean += image_result[rgb][i]
                std += image_result[rgb][i+1]

            mean /= len(image_results)
            std /= len(image_results)

            mean_and_std.append(mean)
            mean_and_std.append(std)

        return mean_and_std
     

if __name__ == '__main__':
    results_folder_path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\results\06-06-2024\\'
    data_generator = DataGenerator('DENV', results_folder_path)
    data_generator.extract_fingerprints_across_files()
    
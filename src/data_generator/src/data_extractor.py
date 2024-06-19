import csv
from glob import glob
from io import StringIO
from statistics import pstdev, mean
from .utils.fingerprint_utils import display_fingerprint, format_line, limit_std

class DataExtractor:
    def __init__(self, sample_type: str, extract_from: str):
        self.sample_type = sample_type
        self.extract_from = extract_from
        self.colors = ['r', 'g', 'b']

    def load_csv_files(self, path: str = None) -> list:
        if path is None:
            path = self.extract_from
        csv_files = glob(path + f"{self.sample_type}*.csv")
        return csv_files
    
    def extract_fingerprint_from_single_csv(self, path: str) -> dict:
        """
        This function reads a csv file containing RGB values and returns a normal distribution of the RGB values.

        csv file format:
        1. The first row contains the column names 
        [   date,time,grid_index ,block_type,
            spot1_r ,spot1_g ,spot1_b ,
            spot2_r ,spot2_g ,spot2_b ,
            bkg_r ,bkg_g ,bkg_b ,
            spot1_corr_r ,spot1_corr_g ,spot1_corr_b ,
            spot2_corr_r ,spot2_corr_g ,spot2_corr_b
        ]
        ...
        
        Args:
            path (str): The path to the csv file
        
        Returns:
            map: A dictionary containing the RGB values of each spot for each block type
        """ 
        
        # read the csv file
        with open(path, 'r') as file:
            data = file.readlines()
        
        # extract the corr RGB values
        rgbs_by_type = self.extract_corr_rgbs(data)

        fingerprint_by_block_type = {}
        for block_type in rgbs_by_type:

            # extract the corr RGB values for each block type
            spot1_corr_rgbs = []
            spot2_corr_rgbs = []
            for block in rgbs_by_type[block_type]:
                spot1_corr_rgbs.append(block[0])
                spot2_corr_rgbs.append(block[1])

            # calculate the mean and standard deviation of the corr RGB values
            block_type_fingerprint = self.get_mean_and_std_of_spots(spot1_corr_rgbs, spot2_corr_rgbs)
            
            # limit the standard deviation to avoid outliers
            block_type_fingerprint = limit_std(block_type_fingerprint, 10)

            fingerprint_by_block_type[block_type] = block_type_fingerprint

        return fingerprint_by_block_type



    def extract_fingerprints_across_files(self, path: str = None, display:bool=False) -> dict:
        """ extract the RGB fingerprint from the csv file"""

        # load the csv files
        csv_files = self.load_csv_files(path)
        
        # extract the RGB fingerprint from each csv file
        fingerprints=[]
        for csv_file in csv_files:
            fingerprints.append(self.extract_fingerprint_from_single_csv(csv_file))

        # append the RGB fingerprints
        appended_fingerprints = self.append_fingerprints(fingerprints)

        # combine the RGB fingerprints
        combined_fingerprints = self.combine_fingerprints(appended_fingerprints)

        if display:
            display_fingerprint(combined_fingerprints)
        
        return combined_fingerprints

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

    def combine_fingerprints(self, appended_fingerprints: dict) -> dict:
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
                block_type_fingerprint[rgb] = self.get_mean_and_std_of_results(appended_fingerprints[block_type], rgb)

            combined_fingerprint[block_type] = block_type_fingerprint

        return combined_fingerprint

    def get_mean_and_std_of_spots(self, spot1_corr_rgbs:list, spot2_corr_rgbs:list)->map:
        """ 
        calculate the mean and standard deviation of the corr RGB values

        Args:
            spot1_corr_rgbs (list): The corr RGB values of spot1
            spot2_corr_rgbs (list): The corr RGB values of spot2

        Format:
            - spot1_corr_rgbs = [[r1, g1, b1], [r2, g2, b2], ...]
            - spot2_corr_rgbs = [[r1, g1, b1], [r2, g2, b2], ...]

        Returns:
            map: A dictionary containing the mean and standard deviation of the RGB values for each spot
        Format: 
            {color: [mean1, std1, mean2, std2]}
        """
        fingerprint = {}
        for i in range(3):
            spot1_rgbs = [rgb[i] for rgb in spot1_corr_rgbs]
            spot2_rgbs = [rgb[i] for rgb in spot2_corr_rgbs]
            fingerprint[self.colors[i]] = [mean(spot1_rgbs), pstdev(spot1_rgbs), mean(spot2_rgbs), pstdev(spot2_rgbs)]
        
        return fingerprint
    
    
    def get_mean_and_std_of_results(self, image_results:list[dict[list[int]]], rgb:str)->list:
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
    
    def extract_corr_rgbs(self, data: list) -> dict: 
        """ extract the corr RGB values from the csv file"""
        rgbs_by_type = {}
        for line in data[1:]:
            line = csv.reader(StringIO(line), delimiter=',', quotechar='"').__next__()
            
            if len(line) < 19:
                continue 
            
            block_type = line[3].strip()
            if block_type not in rgbs_by_type:
                rgbs_by_type[block_type] = []
            
            rgbs_by_type[block_type].append([format_line(line, 0), format_line(line, 1)])
            
        return rgbs_by_type
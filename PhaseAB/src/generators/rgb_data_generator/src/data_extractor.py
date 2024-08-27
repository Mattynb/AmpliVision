""" 
This module contains the DataExtractor class, which is used to extract RGB 
data from CSV files and convert it to fingerprint data.
"""

import csv
from glob import glob
from io import StringIO
from statistics import pstdev, mean

from .utils import Utils


class DataExtractor:
    """
    A class to extract RGB data from CSV files and convert it to fingerprint
    data.

    Attributes:
        sample_type (str): The type of sample (e.g., DENV, ZIKV, CHIKV).
        extract_from (str): The path to the CSV files.
        colors (list): The colors used in the RGB space.
    """

    def __init__(self, sample_type: str, extract_from: str):
        self._sample_type = sample_type  # e.g. DENV, ZIKV, CHIKV
        self._extract_from = extract_from  # path to the csv files
        self._colors = ['r', 'g', 'b']  # color space

    def load_csv_files(
        self,
        path: str = None,
        display: bool = False
    ) -> list[str]:
        """ Load the csv files from the specified path."""

        # if path is not specified, use the default path
        if path is None:
            path = self._extract_from
        csv_files = glob(path + f"{self._sample_type}*.csv")

        if display:
            print(f"Extracting fingerprints from {len(csv_files)} csv files")

        return csv_files

    def convert_spots_to_fingerprints(
        self,
        rgbs_by_type: list[dict[list[list[float]]]]
    ) -> dict[list[list[float]]]:
        """Converts the RGB spots to fingerprints."""

        # puts it into a nicer format to work with
        appended_spots = self.append_spots(rgbs_by_type)

        fingerprint_by_block_type = {}
        for block_type in appended_spots:

            # separate the corr RGB values for each block type
            spot1, spot2 = self.separate_spots(appended_spots[block_type])

            
            # calculate the mean and standard deviation of the corr RGB values
            block_type_fingerprint = self.get_mean_and_std_of_spots(
                spot1, spot2)

        
            # make sure the std is at least 10
            block_type_fingerprint = Utils.ensure_std_floor(
                block_type_fingerprint, 5)

            # add fingerprint to dictionary
            fingerprint_by_block_type[block_type] = block_type_fingerprint

        return fingerprint_by_block_type

    def separate_spots(
        self,
        blocks: list[list[float]]
    ) -> tuple[list[list[float]]]:
        """Separate the corr RGB values of spot1 and spot2 from the blocks."""
        spot1_corr_rgbs = []
        spot2_corr_rgbs = []
        for block in blocks:
            spot1_corr_rgbs.append(block[:3])
            spot2_corr_rgbs.append(block[3:])
        return spot1_corr_rgbs, spot2_corr_rgbs

    def extract(
        self,
        path: str = None,
        display: bool = False
    ) -> dict[str, list[list[float]]]:
        """ extract the RGB fingerprint from the csv file"""

        if path is None:
            path = self._extract_from

        # load the csv files
        csv_files = self.load_csv_files(path)

        # extract the RGB spots from each csv file
        spots = self.extract_spots_from_multiple_csv(csv_files)

        # turn spot into fingerprint
        fingerprints = self.convert_spots_to_fingerprints(spots)

        if display:
            Utils.display_fingerprint(fingerprints)

        return fingerprints

    def append_spots(
        self,
        rgbs_by_type: list[dict[list[list[float]]]]
    ) -> dict[list[list[float]]]:
        """ Append the RGB spots to a map, separated by block type."""
        # initializing map with empty lists
        appended_spots = {
            type_: [] for image in rgbs_by_type for type_ in image
        }

        # appending spots
        for image in rgbs_by_type:
            for type_ in image:
                for block in image[type_]:
                    r1, g1, b1 = block[0]
                    r2, g2, b2 = block[1]
                    appended_spots[type_].append([r1, g1, b1, r2, g2, b2])
        return appended_spots

    def extract_spots_from_multiple_csv(
        self,
        csv_files: list[str]
    ) -> list[dict[list[list[float]]]]:
        """ extract the RGB spots from multiple csv files"""
        spots = []
        for csv_file in csv_files:
            rgbs_by_type = self.extract_spots_from_single_csv(csv_file)
            spots.append(rgbs_by_type)
        return spots

    def extract_spots_from_single_csv(
        self,
        path: str
    ) -> dict[list[list[float]]] | dict[None]:
        """
        This function reads a csv file containing RGB values and returns a 
        normal distribution of the RGB values.

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
            ex: {'block_type1': [ [r1, g1, b1, r2, g2, b2], ..., ] ...more block_types}

        """

        # read the csv file
        try:
            with open(path, 'r', encoding='UTF-8') as file:
                data = file.readlines()
        except FileNotFoundError:
            print(f"File not found: {path}")
            return {}

        # extract the corr RGB values
        rgbs_by_type = self.extract_corr_rgbs(data)

        return rgbs_by_type

    def get_mean_and_std_of_spots(
        self,
        spot1_corr_rgbs: list[list[float]],
        spot2_corr_rgbs: list[list[float]]
    ) -> dict[str, list[float]]:
        """ 
        calculate the mean and standard deviation of the corr RGB values

        Args:
            spot1_corr_rgbs (list): The corr RGB values of spot1
            spot2_corr_rgbs (list): The corr RGB values of spot2

        Format:
            - spot1_corr_rgbs = [[r1, g1, b1], [r2, g2, b2], ...]
            - spot2_corr_rgbs = [[r1, g1, b1], [r2, g2, b2], ...]

        Returns:
            map: A dictionary containing the mean and standard deviation of 
            the RGB values for each spot
        Format: 
            {color: [mean1, std1, mean2, std2]}
        """
        fingerprint = {}
        for i in range(3):
            spot1_rgbs = [rgb[i] for rgb in spot1_corr_rgbs]
            spot2_rgbs = [rgb[i] for rgb in spot2_corr_rgbs]
            fingerprint[self._colors[i]] = [
                mean(spot1_rgbs),
                pstdev(spot1_rgbs),
                mean(spot2_rgbs),
                pstdev(spot2_rgbs)
            ]

        return fingerprint

    def extract_corr_rgbs(
        self,
        data: list[str]
    ) -> dict[str, list[list[float]]]:
        """Extract the corr RGB values from the csv file of each block type."""
        rgbs_by_type = {}
        for line in data[1:]:
            rgbs_by_type = self._extract_corr_rgb_by_type(line, rgbs_by_type)

        return rgbs_by_type

    def _extract_corr_rgb_by_type(
        self,
        line: str,
        rgbs_by_type: dict[str, list[list[float]]]
    ) -> dict[str, list[list[float]]]:
        """Extract the corr RGB values for a specific block type from a single line of CSV data."""
        # Gets one row from csv
        row = next(csv.reader(StringIO(line), delimiter=',', quotechar='"'))

        # skips spaces between data rows
        if len(row) < 19:
            return rgbs_by_type

        # gets block type from row
        block_type = row[3].strip()
        if block_type not in rgbs_by_type:
            # then initializes an empty list for block type if not already there
            rgbs_by_type[block_type] = []

        # appends a list with data -> [corr_r1, corr_g1, corr_b1, corr_r2, corr_g2, corr_b2]
        rgbs_by_type[block_type].append(
            [
                Utils.format_line(row, 0, get_corr=False), 
                Utils.format_line(row, 1, get_corr=False)
            ]
        )

        return rgbs_by_type

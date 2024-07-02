"""
This module contains utility functions for the data_generator module.
"""

import enum
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.stats import norm



class Utils:
    """ Utility functions for the data_generator module. """

    COLORS = ['r', 'g', 'b']

    @staticmethod
    def visualize_generated_pts(data: dict[list[float]], title:str, subtract: int = 0) -> None:
        """
        Visualize generated RGB data points in a grid format.

        Args:
            data (dict[list[float]]): Dictionary containing RGB data points.
            subtract (int, optional): Value to subtract from each RGB component. Defaults to 0.
        """

        GRID_WIDTH = 4

        _, ax = plt.subplots()
        num_types = len(data)
        num_pairs = max(len(pairs) for pairs in data.values())
        grid_height = (num_pairs // GRID_WIDTH) + 1

        for i, (dtype, pairs) in enumerate(data.items()):
            for j, pair in enumerate(pairs):
                row = j // GRID_WIDTH
                col = j % GRID_WIDTH

                data_list = [subtract - x if subtract !=
                             0 else x for x in pair]
                r1, g1, b1, r2, g2, b2 = data_list

                ax.add_patch(plt.Rectangle(
                    (col * 2, i * grid_height + row), 1, 1, color=[r1/255, g1/255, b1/255]))
                ax.add_patch(plt.Rectangle(
                    (col * 2 + 1, i * grid_height + row), 1, 1, color=[r2/255, g2/255, b2/255]))

        ax.set_xlim(-0.1, GRID_WIDTH * 2)
        ax.set_ylim(-0.1, num_types * grid_height)
        ax.set_yticks(np.arange(num_types * grid_height) + 0.5)
        ax.set_yticklabels([dtype for dtype in data.keys()
                            for _ in range(grid_height)])
        ax.set_xticks([])
        plt.title(title)

        plt.show()

    @staticmethod
    def print_generated_pts(pts: dict[list[float]], subtract: int = 0) -> None:
        """
        Print generated RGB data points.

        Args:
            pts (dict[list[float]]): Dictionary containing RGB data points.
            subtract (int, optional): Value to subtract from each RGB component. Defaults to 0.
        """
        print("\n", "-"*10, "GENERATED DATA", "-"*10)
        for dtype, rows in pts.items():
            print(f"{dtype}: ")
            for row in rows:
                row = [subtract - x if subtract != 0 else x for x in row]
                print(f"\t{row}")
        print("-"*30)

    @staticmethod
    def plot_3d_generated_pts(pts: dict[list[float]], title: str = "") -> None:
        """
        Plot generated RGB data points in a 3D space.

        Args:
            pts (dict[list[float]]): Dictionary containing RGB data points.
            title (str, optional): Title of the plot. Defaults to "".
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)

        markers = ['o', 's',  'h', 'D']
        colors = ['r', 'g', 'b', 'c']
        seen_types = set()
        legend_handles = []

        for i, dtype in enumerate(pts):
            for row in pts[dtype]:
                r1, g1, b1, r2, g2, b2 = row
                ax.scatter(r1, g1, b1, color=[
                    r1/255, g1/255, b1/255], s=100, marker=markers[0], edgecolors=colors[i])
                ax.scatter(r2, g2, b2, color=[
                    r2/255, g2/255, b2/255], s=100, marker=markers[1], edgecolors=colors[i])

                if dtype not in seen_types:
                    seen_types.add(dtype)
                    legend_handles.append(
                        plt.Line2D(
                            [0], [0],
                            marker=markers[0],
                            color=colors[i],
                            markerfacecolor='w',
                            markersize=5,
                            label=f'{dtype} spot1'
                        )
                    )
                    legend_handles.append(
                        plt.Line2D(
                            [0], [0],
                            marker=markers[1],
                            color=colors[i],
                            markerfacecolor='w',
                            markersize=5,
                            label=f'{dtype} spot2'
                        )
                    )

        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.legend(handles=legend_handles, loc='best')
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')

        plt.show()

    @classmethod
    def generate_gendata_csv_filename(cls, sample_type: str, og_folder_path: str) -> str:
        """
        Generate a filename for the generated CSV file.

        Args:
            id (int): Identifier for the filename.
            og_folder_path (str): Original folder path.

        Returns:
            str: Generated filename.
        """
        image_name = cls.get_filename(0, og_folder_path)
        image_name = image_name[:image_name.find("_")]

        return f"/{sample_type}_generated.csv"

    @staticmethod
    def get_filename(_id: int, path: str) -> str:
        """
        Get the filename from a given path.

        Args:
            id (int): Identifier for the file.
            path (str): Path to the directory or file.

        Returns:
            str: Extracted filename.
        """
        # Check if path is a directory
        if path.endswith('\\') or path.endswith('*'):
            path = path.removesuffix("*")
            files = [file for file in os.listdir(
                path) if os.path.isfile(os.path.join(path, file))]
            image_name = files[_id-1].replace('.', '_')

        # or if path is a file
        else:
            start_i = path.rfind("\\")
            image_name = path[start_i + 1:]

        return image_name

    @staticmethod
    def create_rows(data: dict[list[list[int]]], classification:bool = None) -> list:
        """
        Create rows of data for CSV writing.

        Args:
            data (dict[list[list[int]]]): Dictionary containing RGB data points.

        Returns:
            list: List of rows to be written to the CSV.
        """
        rows = []
        for i, (b_type, corr_rgbs) in enumerate(data.items()):
            for rgb in corr_rgbs:
                row = [b_type] + rgb

                if classification:
                    row.append(i)
                
                rows.append(row)
        return rows

    @classmethod
    def write_to_csv(
        cls, 
        sample_type: str, 
        folder_path: str, 
        data: dict[list[list[int]]] = None,
        rows: list[str] = None,
        classification: bool = False
    ) -> None:
    
        """
        Write RGB data points to a CSV file.

        Args:
            folder_path (str): Path to the folder where the CSV will be saved.
            data (dict[list[list[int]]]): Dictionary containing RGB data points.
        """

        if data is None and rows is None:
            raise Exception(
                "Must pass in either \'rows\' or \'data\' parameter when using Utils.write_to_csv"
            )
        
        filename = cls.generate_gendata_csv_filename(sample_type, folder_path)

        now = datetime.now()
        date = now.strftime("%m-%d-%Y")
        subfolder_name = f'/{date}'

        output_folder = folder_path + subfolder_name
        os.makedirs(output_folder, exist_ok=True)
        output = output_folder + filename
    
        if rows is None:
            _rows = cls.create_rows(data)
        else:
            _rows = rows

        with open(
            output,
            'w',
            encoding='utf-8',
            newline=''
        ) as csvfile:

            csvwriter = csv.writer(csvfile)
            headers = ['block_type', 'spot1_corr_r', 'spot1_corr_g',
                       'spot1_corr_b', 'spot2_corr_r', 'spot2_corr_g', 'spot2_corr_b']

            if classification:
                headers.insert(0,'class')

            csvwriter.writerow(headers)
            csvwriter.writerows(_rows)


    @classmethod
    def combine_generated_cvs(cls, folder_path):
        
        now = datetime.now()
        date = now.strftime("%m-%d-%Y")
 
        combined_data = cls.combine_csvs_data(folder_path + f'/{date}')
        
        cls.write_to_csv(
            "COMBINED", 
            folder_path, 
            rows=combined_data,
            classification=True
        )

    @staticmethod
    def combine_csvs_data(folder_path : str) -> list[str]:
        # get file names
        csv_files = [
            file for file in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, file))
        ]

        if "COMBINED_generated.csv" in csv_files:
            csv_files.remove('COMBINED_generated.csv')

        _data = []
        for i, file in enumerate(csv_files):

            file_path = folder_path + '\\' + file
            # read the csv file
            try:
                with open(file_path, 'r', encoding='UTF-8') as file:
                    data = file.readlines()
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return {}

            _buffer = [str(i) + ',' + bi for bi in data[1:]]
            _data.append(_buffer)

        combined_data = []
        for test in _data:
            for block in test:
                combined_data.append(block.strip('\n').split(',')) 
        
        return combined_data
 
    @staticmethod
    def subtract_255(rgb: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        """
        Subtract 255 from each RGB value in the given dictionary.

        Args:
            rgb (dict[str, list[list[int]]]): Dictionary containing RGB data points.

        Returns:
            dict[str, list[list[int]]]: Dictionary with 255 subtracted from each RGB value.
        """
        return {
            block_type: [
                [255 - x for x in row] for row in rgb_list] for block_type, rgb_list in rgb.items()
        }

    @classmethod
    def limit_std(
        cls,
        fingerprint: dict[str, dict[str, list[float]]],
        limit: int = 15
    ) -> dict[str, dict[str, list[float]]]:
        """
        Limit the standard deviation values in the fingerprint dictionary.

        Args:
            fingerprint (dict[str, dict[str, list[float]]]): Dictionary containing fingerprint data.
            limit (int, optional): The maximum allowed standard deviation. Defaults to 15.

        Returns:
            dict[str, dict[str, list[float]]]: Dictionary with limited standard deviation values.
        """
        for color in cls.COLORS:
            for i in range(1, 4, 2):
                if fingerprint[color][i] > limit:
                    fingerprint[color][i] = limit
        return fingerprint

    @staticmethod
    def format_line(line: list[str], spot: int) -> list[float]:
        """
        Format a line from the data file.

        Args:
            line (list[str]): The line to format.
            spot (int): The spot index.

        Returns:
            list[float]: The formatted line as a list of floats.
        """
        base_index = spot * 3
        return [
            float(line[base_index + 13].strip()),  # corr_spotX_r
            float(line[base_index + 14].strip()),  # corr_spotX_g
            float(line[base_index + 15].strip())   # corr_spotX_b
        ]

    @classmethod
    def display_fingerprint(
        cls,
        fingerprint: dict[str, dict[str, list[float]]],
        title_sufx: str = ''
    ) -> None:
        """
        Display fingerprint data as distribution plots.

        Args:
            fingerprint (dict[str, dict[str, list[float]]]): Dictionary containing fingerprint data.
            title_sufx (str, optional): Suffix for the plot title. Defaults to ''.
        """
        for block_type, values in fingerprint.items():
            title = f'{block_type} fingerprint {title_sufx}'
            _, ax = plt.subplots(3, 1, figsize=(10, 10), num=title)

            for i, color in enumerate(cls.COLORS):
                mean1, std1, mean2, std2 = values[color]

                if std1 == 0 and std2 == 0:
                    print(f'No std for {color} in {block_type}, using random')
                    std1, std2 = random.randint(1, 2), random.randint(1, 2)

                x = range(0, 256)
                ax[i].plot(x, norm.pdf(x, mean1, std1),
                           color=color, label='spot1')
                ax[i].plot(x, norm.pdf(x, mean2, std2), color=color,
                           linestyle='dashed', label='spot2')
                ax[i].set_title(f'{color} distribution')
                ax[i].legend()
            plt.show()

    @staticmethod
    def visualize_fingerprints_with_colors(fingerprints: dict[str, dict[str, list[float]]]) -> None:
        """
        Visualize fingerprints with actual colors.

        Args:
            fingerprints (dict[str, dict[str, list[float]]]): 
               |__ Dictionary containing fingerprint data.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        fig.suptitle(
            'RGB Values for Different Blocks with Actual Colors', fontsize=16)

        block_names = list(fingerprints.keys())

        for _, (block, ax) in enumerate(zip(block_names, axes.flatten())):
            data = fingerprints[block]
            x_labels = ['Spot 1', 'Spot 2']

            for i in range(2):
                r_mean = data['r'][i * 2]
                g_mean = data['g'][i * 2]
                b_mean = data['b'][i * 2]
                r_stdev = data['r'][i * 2 + 1]
                g_stdev = data['g'][i * 2 + 1]
                b_stdev = data['b'][i * 2 + 1]

                color = [r_mean / 255, g_mean / 255, b_mean / 255]
                
                ax.errorbar(x_labels[i], r_mean, yerr=r_stdev,
                            fmt='o', color='r', capsize=20)
                ax.errorbar(x_labels[i], g_mean, yerr=g_stdev,
                            fmt='o', color='g', capsize=20)
                ax.errorbar(x_labels[i], b_mean, yerr=b_stdev,
                            fmt='o', color='b', capsize=20)

                ax.scatter([x_labels[i]], [r_mean], color=color,
                           s=100, marker='D', edgecolor='r', zorder=3)
                ax.scatter([x_labels[i]], [g_mean], color=color,
                           s=100, marker='D', edgecolor='g', zorder=3)
                ax.scatter([x_labels[i]], [b_mean], color=color,
                           s=100, marker='D', edgecolor='b', zorder=3)

            ax.set_title(block)
            ax.set_ylim(0, 255)
            ax.set_ylabel('Value')
            ax.legend(['R Mean ± Std', 'G Mean ± Std', 'B Mean ± Std'])
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == '__main__':

    path = r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\generated_results\\"
    Utils.combine_generated_cvs(path)
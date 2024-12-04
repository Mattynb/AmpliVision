"""
This module contains utility functions for the data_generator module.
"""

import os
import csv
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.stats import norm


class Utils:
    """ Utility functions for the data_generator module. """

    COLORS = ['r', 'g', 'b']

    @classmethod
    def comb_pts_as_flat_json(cls, comb_pts: dict[list[float]], class_n: int, display: bool = 0) -> list:

        row = []
        columns = ['class']
        block_types = ['Test Block 1', 'Test Block 2',
                       'Test Block 3', 'Control Block']

        # create column names
        for b_type in block_types:
            for s in range(1, 3):  # spot 1 and spot 2
                columns.append(f'{b_type}_spot{s}_r')
                columns.append(f'{b_type}_spot{s}_g')
                columns.append(f'{b_type}_spot{s}_b')
        row.append(columns)

        # create rows
        for i in range(len(comb_pts['Test Block 1'])):
            row.append(
                [class_n]  # which disease it is
                +
                [   # the rgb data for each spot in each block type
                    comb_pts[b_type][i][j]
                    for b_type in block_types
                    for j in range(6)
                ]
            )

        # display the data
        if display:
            for r in row:
                print(r)
            print("\n", f"dimension: {len(row)} x {len(row[0])}", "\n")

        return row

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

    @classmethod
    def write_to_csv(
        cls,
        folder_path: str,
        # list of rgb data, inside a list of block types inside a list of rows, inside a list of classes
        data: list[list[list[int]]]
    ) -> str:
        """
        Write RGB data points to a CSV file.

        Args:
            folder_path (str): Path to the folder where the CSV will be saved.
            data (dict[list[list[int]]]): Dictionary containing RGB data points.
        """

        # prepare folder and get the filename
        output = cls.prepare_to_write_csv(folder_path, data)

        # write the data to the csv file
        with open(
            output,
            'w',
            encoding='utf-8',
            newline=''
        ) as csvfile:

            csvwriter = csv.writer(csvfile)

            # write the headers
            headers = data[0][0]
            csvwriter.writerow(headers)

            # write the data
            for row in data:
                csvwriter.writerows(row[1:])

        return output

    @classmethod
    def prepare_to_write_csv(
        cls,
        folder_path: str,
        filename: str,
    ) -> str:
        """
        Write RGB data points to a CSV file.

        Args:
            folder_path (str): Path to the folder where the CSV will be saved.
            data (dict[list[list[int]]]): Dictionary containing RGB data points.
        """

        # get the filename
        filename = cls.generate_gendata_csv_filename("COMBINED", folder_path)

        # get the current date to create a subfolder
        now = datetime.now()
        date = now.strftime("%m-%d-%Y")
        subfolder_name = f'/{date}'

        # create the folder if it doesn't exist
        output_folder = folder_path + subfolder_name
        os.makedirs(output_folder, exist_ok=True)
        output = output_folder + filename

        return output

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
    def ensure_std_ceiling(
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

    @classmethod
    def ensure_std_floor(
        cls,
        fingerprint: dict[str, dict[str, list[float]]],
        floor: int = 2
    ) -> dict[str, dict[str, list[float]]]:
        """
        Fix the standard deviation values in the fingerprint dictionary.

        Args:
            fingerprint (dict[str, dict[str, list[float]]]): Dictionary containing fingerprint data.
            fix (int, optional): The fixed standard deviation. Defaults to 2.

        Returns:
            dict[str, dict[str, list[float]]]: Dictionary with fixed standard deviation values.
        """
        for color in cls.COLORS:
            for i in range(1, 4, 2):
                if fingerprint[color][i] < floor:
                    fingerprint[color][i] = floor
        return fingerprint

    @staticmethod
    def format_line(line: list[str], spot: int, get_corr:bool = False) -> list[float]:
        """
        Format a line from the data file.

        Args:
            line (list[str]): The line to format.
            spot (int): The spot index.

        Returns:
            list[float]: The formatted line as a list of floats.
        """
        base_index = spot * 3
    
        if get_corr: 
            return [
                float(line[base_index + 13].strip()),  # corr_spotX_r
                float(line[base_index + 14].strip()),  # corr_spotX_g
                float(line[base_index + 15].strip())   # corr_spotX_b
            ] 
        
        else:
            return [
                float(line[base_index + 4].strip()),  # spotX_r
                float(line[base_index + 5].strip()),  # spotX_g
                float(line[base_index + 6].strip())   # spotX_b
            ]

    """ ---- Display functions ---- """

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

    @staticmethod
    def visualize_generated_pts(data: dict[list[float]], title: str, subtract: int = 0) -> None:
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
        for btype, rows in pts.items():
            print(f"{btype}: ")
            for row in rows:
                row = [subtract - x if subtract != 0 else x for x in row]
                print(f"\t{row}")
        print("-"*30)

        print("\n", f"dimension: {len(pts)} x {len(rows)}", "\n")

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


if __name__ == '__main__':

    path = ...
    Utils.combine_generated_cvs(path)

from statistics import mean, pstdev
import matplotlib.pyplot as plt
from scipy.stats import norm 
from io import StringIO
import csv


def extract_fingerprint_from_single_csv(path:str)->map:
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
    rgbs_by_type = get_corr_rgbs(data)

    fingerprint_by_block_type = {}
    for block_type in rgbs_by_type:

        # extract the corr RGB values for each block type
        spot1_corr_rgbs = []
        spot2_corr_rgbs = []
        for block in rgbs_by_type[block_type]:
            spot1_corr_rgbs.append(block[0])
            spot2_corr_rgbs.append(block[1])

        # calculate the mean and standard deviation of the corr RGB values
        block_type_fingerprint = get_mean_and_std(spot1_corr_rgbs, spot2_corr_rgbs)
        
        # limit the standard deviation to avoid outliers
        block_type_fingerprint = limit_std(block_type_fingerprint, 10)

        fingerprint_by_block_type[block_type] = block_type_fingerprint

    return fingerprint_by_block_type

def get_corr_rgbs(data:list)->map:
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

def format_line(line:list, spot:int)->list:
    """ format the line to extract the corr RGB values"""
    spot = spot*3
    return [float(line[spot+13].strip()), float(line[spot+14].strip()), float(line[spot+15].strip())]


def get_mean_and_std(spot1_corr_rgbs:list, spot2_corr_rgbs:list)->map:
    """ 
    calculate the mean and standard deviation of the corr RGB values
    
    Args:
        spot1_corr_rgbs (list): The corr RGB values of spot1
        spot2_corr_rgbs (list): The corr RGB values of spot2
    
    Format:: 
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
        fingerprint[colors[i]] = [mean(spot1_rgbs), pstdev(spot1_rgbs), mean(spot2_rgbs), pstdev(spot2_rgbs)]
    

    
    return fingerprint

def limit_std(fingerprint:map, limit:int =15)->map:
    """ make sure the standard deviation is not too high to avoid outliers"""
    for i in range(3):
        for j in range(1, 4, 2):
            if fingerprint[colors[i]][j] > limit:
                fingerprint[colors[i]][j] = limit
    return fingerprint
    

def display_fingerprint(fingerprint:map)->None:
    """
    This function displays the R, G, and B distribution using matplotlib.
    
    Args:
        fingerprint (list): A list of RGB values
    """
    
   

    for block_type in fingerprint:
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), num=f'{block_type} fingerprint')
        # plot structure:
        # row 1 -> R distribution
        # row 2 -> G distribution
        # row 3 -> B distribution






        for i in range(3):
            mean1, std1, mean2, std2 = fingerprint[block_type][colors[i]]
            x = range(0, 256)
            ax[i].plot(x, norm.pdf(x, mean1, std1), color=colors[i], label='spot1')
            ax[i].plot(x, norm.pdf(x, mean2, std2), color=colors[i], linestyle='dashed', label='spot2')
            ax[i].set_title(f'{colors[i]} distribution')
            ax[i].legend()
 
        plt.show()


# global variable to store the colors
colors = ['r', 'g', 'b']

if __name__ == '__main__':
    path = r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\results\06-06-2024\DENV_IMG_62.JPEG_results_06-06-2024_(16-27-23).csv"
    fingerprint = extract_fingerprint_from_single_csv(path)

    # print the fingerprint
    for block_type in fingerprint:
        print(f'{block_type} fingerprint:')
        for color in colors:
            mean1 = fingerprint[block_type][color][0]
            std1 = fingerprint[block_type][color][1]
            mean2 = fingerprint[block_type][color][2]
            std2 = fingerprint[block_type][color][3]
            print(f'\t{color}:\n\t\tspot1: mean=[{mean1}], std=[{std1}]\n\t\tspot2: mean=[{mean2}], std=[{std2}]')
        print('\n')


    #display_fingerprint(fingerprint)
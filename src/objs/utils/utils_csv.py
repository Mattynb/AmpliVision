import os
import csv
from datetime import datetime

def generate_csv_filename(id:int, path:str)->str:
    # get current date and time
    now = datetime.now()

    # get image name
    image_name = get_filename(id, path)
    
    # format date and time
    date = now.strftime("%m-%d-%Y")
    time = now.strftime("(%H-%M-%S)")

    return f"{date}/{image_name}_results_{date}_{time}.csv"


def get_filename(id:int, path:str):
    # if path is a directory
    if path[-1] == '\\' or path[-1] == '*':
        path = path.removesuffix("*")

        # load directory
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        image_name = files[id-1].replace('.', '_') 

    else:
        start_i = path.rfind("\\")
        image_name = path[(start_i + 1):]

    return image_name

def write_to_csv(filename:str, data: list)->None:
    
    # create folder for csv files
    subfolder_name = filename.split('/')[-2]
    if not os.path.exists("data/results/" + subfolder_name):
        os.makedirs("data/results/" + subfolder_name)


    with open("data/results/" + filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        format_str = [
            'date', ' time',
            ' grid_index',
            ' block_type ',
            ' spot1_r', ' spot1_g', ' spot1_b',
            ' spot2_r', ' spot2_g', ' spot2_b',
            ' bkg_r', ' bkg_g', ' bkg_b',
            ' spot1_corr_r', ' spot1_corr_g', ' spot1_corr_b',
            ' spot2_corr_r', ' spot2_corr_g', ' spot2_corr_b',
        ]

        # writing the data
        csvwriter.writerow(format_str)
        csvwriter.writerows(data)
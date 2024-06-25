import os
import csv
from datetime import datetime

def generate_gendata_csv_filename(id:int, og_folder_path:str)->str:
    # get current date and time
    now = datetime.now()

    # get image name
    image_name = get_filename(0, og_folder_path)
    
    # format date and time
    date = now.strftime("%m-%d-%Y")
    time = now.strftime("(%H-%M-%S)")

    image_name = image_name[:image_name.find("_")]
    

    return f"/generated_{id}_{'here'}.csv"


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

def create_rows(data:dict[list[list[int]]])->list:    
    rows = []
    for b_type in data:
        for corr_rgbs in data[b_type]:
            # setting all rgb values to None
            spot1_corr_r, spot1_corr_g, spot1_corr_b = corr_rgbs[:3]
            spot2_corr_r, spot2_corr_g, spot2_corr_b = corr_rgbs[3:]

            # create data to be written to csv
            row = [
                b_type, 
                spot1_corr_r, spot1_corr_g, spot1_corr_b, 
                spot2_corr_r, spot2_corr_g, spot2_corr_b
            ]
            rows.append(row)
    
    return rows



def write_to_csv(folder_path:str, data: list)->None:
    
    filename = generate_gendata_csv_filename(0, folder_path)

    data = create_rows(data)

    # create folder for csv files
    now = datetime.now()
    date = now.strftime("%m-%d-%Y")
    subfolder_name = f'/{date}'
    print("folder: ", folder_path)
    print("subfolder: ", subfolder_name)
    
    if not os.path.exists("data/generated_results/" + subfolder_name):
        os.makedirs("data/generated_results/" + subfolder_name)


    with open("data/generated_results" + subfolder_name + filename, 'w',  newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        format_str = [
            'block_type',
            'spot1_corr_r', 'spot1_corr_g', 'spot1_corr_b',
            'spot2_corr_r', 'spot2_corr_g', 'spot2_corr_b',
        ]

        # writing the data
        csvwriter.writerow(format_str)
        for row in data:
            csvwriter.writerow(row)


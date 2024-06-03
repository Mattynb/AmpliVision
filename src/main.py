from objs import Grid, GridImageNormalizer, ImageLoader, ColorContourExtractor, TestAnalyzer
from backend import identify_block

import time
import csv
import cv2 as cv
from datetime import datetime
import os


def main(path_to_imgs):
    """
    ### Main function
    ---------------
    Main function of the program. Loads the images, creates the Image objects, and finds the blocks in the image.

    #### Args:
    path_to_imgs: path to images to be loaded

    #### Returns:
    None
    """

    # loading images from given path
    images = ImageLoader.load_images(path_to_imgs) #"image0_scaned.jpg")
    
    # Analyzing each image
    for id, image in enumerate(images):

        ## display
        
        #   Create Image object from loaded image.
        # The Image object is used to store the image 
        # and the steps of the image processing.
        t = time.time()
        image_scan = GridImageNormalizer.scan(id, image); print("scanned in: ", round(time.time()-t,2), "s")

        #image_scan = image
        if image_scan is None: continue

        display(image_scan, 1000, 'I guess its u')
        

        # When working with repeat image, uncomment the line below 
        # and comment the lines above   
        cv.imwrite(f"image{id}_scaned.jpg", image_scan)
    

        #   Finds the contours around non-grayscale (colorful) 
        # edges in image. The contours are used to find the 
        # pins and later blocks.
        contours = ColorContourExtractor.process_image(image_scan)

        #   Create Grid object from the scanned image. The grid
        # is used to store information about the grid, such as 
        # the blocks and pins, etc.
        Grid_DS = Grid(image_scan)

        # determines what squares in grid are
        #  blocks
        Grid_DS.find_blocks(contours); print(f"there are {len(Grid_DS.blocks)} blocks in the grid")

        """## display
        im = Grid_DS.img.copy()
        Grid_DS.draw_gridLines(im)
        for block in Grid_DS.get_blocks():
            block.draw_test_area(im)
        display(im, 0)"""

        # identifies type of blocks in the grid
        csv_filename = generate_csv_filename(id, path_to_imgs)
        csv_rows = []
        for block in Grid_DS.get_blocks():
            block.set_rgb_sequence()
            block = identify_block(block)

            # analyse results of test blocks
            if block.get_block_type() == "Test Block":
                print(f"Test block found. Analyzing block and exporting to {csv_filename}")
                ta = TestAnalyzer(block)
                csv_rows.append(ta.analyze_test_result())

        write_to_csv(csv_filename, csv_rows)

def generate_csv_filename(id:int, path:str):
    # get current date and time
    now = datetime.now()

    # if path is a directory
    if path[-1] == '\\' or path[-1] == '*':
        path = path.removesuffix("*")

        # load directory
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        image_name = files[id].replace('.', '_') 

    else:
        start_i = path.rfind("\\")
        image_name = path[(start_i + 1):]
    
    # format date and time
    date = now.strftime("%m-%d-%Y")
    time = now.strftime("(%H-%M-%S)")

    return f"{image_name}_results_{date}_{time}.csv"

def write_to_csv(filename:str, data: list)->None:
    with open("data/results/" + filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        format_str = [
            'date', ' time',
            ' grid_index',
            ' block_type ',
            ' bkg_r', ' bkg_g', ' bkg_b',
            ' test_r', ' test_g', ' test_b',
            ' cntrl_r', ' cntrl_g', ' cntrl_b'
        ]

        # writing the data
        csvwriter.writerow(format_str)
        csvwriter.writerows(data)

def display(image, t=100, title= 'image'):
    im = cv.resize(image, (0,0), fx=0.5, fy=0.5)
    cv.imshow(title, im)
    cv.waitKey(t)
    cv.destroyAllWindows()

if __name__ == '__main__':
    path_to_imgs = r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\New_images_051524\IMG_6066.jpg" #*"
    main(path_to_imgs)
    #print(generate_csv_filename(0, path_to_imgs))




"""
TODO:
actually use white balancing. Curently it is being called on image_scanner but not being used.

TODO:
write on saved image the identified block types, and the sequence of the block

TODO:
add reference to scan

TODO: 
image generation with blocks for U-NET

TODO:
make a block class that inherits from the square class?
"""
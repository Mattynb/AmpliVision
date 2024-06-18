from objs import Grid, GridImageNormalizer, ImageLoader, ColorContourExtractor, TestAnalyzer
from objs.utils import generate_csv_filename, write_to_csv, get_filename
from backend import identify_block

import sys 
import time
import cv2 as cv


def main(path_to_imgs:str)->None:
    """
    ### Main function
    ---------------
    Main function of the program. Loads the images, creates the Image objects, and finds the blocks in the image.
    Then it identifies the type of blocks in the image and exports the results to a csv file.

    #### Args:
    path_to_imgs: path to images to be loaded

    #### Returns:
    None
    """

    # loading images from given path
    images = ImageLoader.load_images(path_to_imgs) #"image0_scaned.jpg")
    
    # for display
    print(f"Images to be analyzed: {len(images)}\n")
    print(type(images[0]))
    
    # Analyzing each image
    for id, image in enumerate(images):

        #   Create Image object from loaded image.
        # The Image object is used to store the image 
        # and the steps of the image processing.
        t = time.time() 
        image_name = get_filename(id, path_to_imgs)
        image_scan = GridImageNormalizer.scan(image_name, image); print("scanned in: ", round(time.time()-t,2), "s\n")

        #image_scan = image
        if image_scan is None: continue

        #display(image_scan, 1000, 'I guess its u')
        

        # When working with repeat image, uncomment the line below 
        # and comment the lines above   
        #cv.imwrite(f"{image_name}_scaned.jpg", image_scan)
    
        #   Finds the contours around non-grayscale (colorful) 
        # edges in image. The contours are used to find the 
        # pins and later blocks.
        contours = ColorContourExtractor.process_image(image_scan)

      
        # display
        #"""
        im = image_scan.copy()
        #Grid_DS.draw_gridLines(im)
        #for block in Grid_DS.get_blocks():
        #    block.draw_pins(im)
        im = cv.drawContours(im, contours, -1, (0,255,0), 3)
        display(im, 0)
        
        #"""
        #   Create Grid object from the scanned image. The grid
        # is used to store information about the grid, such as 
        # the blocks and pins, etc.
        Grid_DS = Grid(image_scan)

        # determines what squares in grid are
        #  blocks
        Grid_DS.find_blocks(contours); print(f"there are {len(Grid_DS.blocks)} blocks in the grid")

        # identifies type of blocks in the grid
        csv_filename = generate_csv_filename(id, path_to_imgs)
        csv_rows = []
        for block in Grid_DS.get_blocks():
            block.set_rgb_sequence()
            block = identify_block(block)

            # analyse results of test blocks
            if block.get_block_type() in ("Test Block", "Control Block"):
                print(f"{block.get_block_type()} found. Analyzing block and exporting to {csv_filename}")
                ta = TestAnalyzer(block)
                csv_rows.append(ta.analyze_test_result())

        write_to_csv(csv_filename, csv_rows)

def display(image, t=100, title= 'image'):
    im = cv.resize(image, (0,0), fx=0.5, fy=0.5)
    cv.imshow(title, im)
    cv.waitKey(t)
    cv.destroyAllWindows()

if __name__ == '__main__':
    # take path as command line argument or use default path
    """if len(sys.argv) == 2:
        path_to_imgs = rf"{sys.argv[1]}"
        print("path given in command line: ", path_to_imgs)

    else:
        path_to_imgs = r"app\\data\\New_images_060324\\IMG_6159.JPEG" #*"
        print("path not given in command line. Using default path: ", path_to_imgs)
    # """
    path_to_imgs = r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\New_images_060324\IMG_6159.JPEG"

    main(path_to_imgs)



"""
TODO:
actually use white balancing. Curently it is being called on image_scanner but not being used.

TODO:
write on saved image the identified block types, and the sequence of the block

TODO:
add reference to scan

TODO: 
image generation with blocks for U-NET
"""
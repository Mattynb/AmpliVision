import os
import cv2 as cv

from .objs.utils import get_filename
from .objs import Grid, GridImageNormalizer, ImageLoader, ColorContourExtractor


def phaseA1(path_to_imgs: str, display: bool = False, pre_scanned: bool = False) -> tuple:
    """
    Phase A
    ---------------
    Main function of the program. Loads the images, creates the Image objects, and finds the blocks in the image.
    Then it identifies the type of blocks in the image and exports the results to a csv file.

    #### Args:
    path_to_imgs: path to images to be loaded

    #### Returns:
    Grid_DS: Grid object with the blocks and pins
    image_name: name of the image
    """

    # loading images from given path
    images = ImageLoader.load_images(path_to_imgs)
    print(f"Images to be analyzed: {len(images)}\n")

    # Analyzing each image
    Grids = {}
    for idx, image in enumerate(images):

        # skip seen files
        if seen_file(idx):
            continue

        image_scan, image_name = get_image_scan(image, pre_scanned)

        # skip if unable to scan image
        if image_scan is None:
            continue

        #   Finds the contours around non-grayscale (colorful)
        # edges in image. The contours are used to find the
        # pins and later blocks.
        contours = ColorContourExtractor.process_image(image_scan)

        #   Create Grid object from the scanned image. The grid
        # is used to store information about the grid, such as
        # the blocks and pins, etc.
        Grid_DS = Grid(image_scan)
        display_grid(image_scan, Grid_DS, contours) if display else None

        # determines which squares in grid are blocks
        Grid_DS.find_blocks(contours)
        print(f"there are {len(Grid_DS.blocks)} blocks in the grid")

        # saves the grid object to a dictionary
        Grids[image_name] = Grid_DS

    return Grids

# ----------------- Helper Functions ----------------- #


def get_image_scan(image, pre_scanned, idx, path_to_imgs):
    """Create Image object from loaded image.
    The Image object is used to store the image
    and the steps of the image processing."""

    if pre_scanned:
        image_scan = image
    else:
        image_name = get_filename(idx, path_to_imgs)
        image_scan = GridImageNormalizer.scan(image_name, image)
        cv.imwrite(f"scanned/{image_name}_scaned.jpg", image_scan)

    return image_scan, image_name


def display(image, t=100, title='image'):
    im = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv.imshow(title, im)
    cv.waitKey(t)
    cv.destroyAllWindows()


def display_grid(image_scan, Grid_DS, contours):
    im = image_scan.copy()
    Grid_DS.draw_gridLines(im)
    for block in Grid_DS.get_blocks():
        block.draw_pins(im)
    im = cv.drawContours(im, contours, -1, (0, 255, 0), 3)
    display(im, 0)


def seen_file(idx: int) -> bool:
    path = r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\scanned"
    if idx < len(os.listdir(path)) - 1:
        return True
    return False

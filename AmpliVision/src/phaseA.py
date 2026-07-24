import os
import cv2 as cv
from numpy import ndarray

from .objs.utils import get_filename_safe
from .objs import Grid, GridImageNormalizer, ImageLoader, ColorContourExtractor, TestGraph


def phaseA1(
    path_to_imgs: str,
    scanned_path: str,
    display: bool = False,
    is_pre_scanned: bool = False,
    do_white_balance: bool = False,
    file_paths: list = None,
) -> list:
    """
    Phase A1
    ---------------
    Loads and scans the images.

    #### Args:
    path_to_imgs: path to images to be loaded

    #### Returns:
    scanned_images : The images scanned
    """

    if path_to_imgs.startswith(scanned_path):
        is_pre_scanned = True

    paths = file_paths if file_paths is not None else ImageLoader.load_images(
        path_to_imgs, return_paths_only=True
    )
    images = ImageLoader.load_images(path_to_imgs, file_paths=paths)
    print(f"# of Images to be analyzed: {len(images)}\n")

    scanned_images = {}
    for image, src_path in zip(images, paths):
        image_name = get_filename_safe(src_path)

        if not is_pre_scanned and seen_file(image_name, scanned_path):
            continue

        image_scan, image_name = get_image_scan(
            image, is_pre_scanned, image_name, scanned_path, white_balance=do_white_balance
        )

        if image_scan is None:
            print(f"Unable to scan image {path_to_imgs}{image_name}, skipping...")
            continue

        scanned_images[image_name] = image_scan
        if display:
            display_img(image_scan, 0, image_name)

    return scanned_images


def phaseA2(scanned_images: dict, display: bool = False) -> dict:
    """
    Phase A2
    ---------------
    Finds the blocks in the image and creates the virtual representation of grids.

    #### Args:
    scanned_images: the scanned images

    #### Returns:
    Grids: Grid object with the blocks and pins
    """

    Grids = {}
    for image_name, image_scan in scanned_images.items():

        print(f"Building \'{image_name}\''s virtual grid...") if display else None

        #   Finds the contours around non-grayscale (colorful)
        # edges in image. The contours are used to find the
        # pins and later blocks.
        contours = ColorContourExtractor.process_image(image_scan, display=display)

        #   Create Grid object from the scanned image. The grid
        # is used to store information about the grid, such as
        # the blocks and pins, etc.
        Grid_DS = Grid(image_scan)
        #display_grid(image_scan, Grid_DS, contours) if display else None

        # determines which squares in grid are blocks
        Grid_DS.find_blocks(contours)

        # saves the grid object to a dictionary
        Grids[image_name] = Grid_DS

    return Grids


def phaseA3(Grids, display: bool = False):
    """
    Phase A2
    ---------------
    Generates graph like structures of the tests, aka what is the sequence of blocks being used.

    #### Args:
    Grids: Grid object with the blocks and pins

    #### Returns:
    graphs: Position Graph representing the configuration of the test blocks. Eg. Sample -> Test Block 1 -> etc.
    """
    graphs = []
    for _, grid in Grids.items():

        blocks = grid.get_blocks()

        TG = TestGraph(blocks)
        graphs.append(TG.graph)

        # Draw the graph
        if display:
            TG.display()

    return graphs


# ----------------- Helper Functions ----------------- #

def get_image_scan(image, is_pre_scanned: bool, image_name: str, scanned_path: str, white_balance: bool) -> tuple:
    if is_pre_scanned:
        image_scan = image
    else:
        image_scan = GridImageNormalizer.scan(image_name, image, do_white_balance=white_balance)
        cv.imwrite(f"{scanned_path}{image_name}_scaned.jpg", image_scan)
    return image_scan, image_name


def display_img(image, t=100, title='image'):
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
    display_img(im, 0)

def seen_file(image_name: str, scanned_path: str) -> bool:
    " Check whether a scanned version of this specific image already exists. "
    expected_path = f"{scanned_path}{image_name}_scaned.jpg"
    return os.path.exists(expected_path)
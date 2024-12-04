import math
import itertools
import cv2 as cv
import numpy as np

from ..utils import Utils
from .Square import Square
from ..utils.utils_geometry import is_arranged_as_square, find_center_of_points, find_center_of_contour


class Grid:
    def __init__(self, img: np.ndarray):
        # scanned image
        self.img = img.copy()

        # setup ratios used in the grid
        # such as the plus minus, etc.
        self.setup_ratios()

        # saving blocks here and in grid creates 2 sources of truth.
        # This list should keep index or something.
        self.blocks = []

        # represents the grid in the image as a 2D array of squares. Initialized as 2D array of None to represent empty.
        self.grid = [
            [None for _ in range(self.MAX_INDEX + 1)]
            for _ in range(self.MAX_INDEX + 1)
        ]

        self.create_grid()

    ## Setup functions ##
    def setup_ratios(self):
        """
        ### Setup rations
        Function that sets up the ratios used in the grid.
        """
        # max x and y coordinates of the image
        self.MAX_XY = self.img.shape[0]  # assumes image is square

        # ratios measured experimentally as a percentage of the grid
        # size of pin diameter/grid size
        self.PIN_RATIO = int(self.MAX_XY * 0.012)
        # size of edge/grid size. Edge is the "lines" around squares
        self.EDGE_RATIO = int(self.MAX_XY * 0.01)
        # squares are the places where you can insert the ampli blocks
        self.SQUARE_RATIO = int(self.MAX_XY * 0.089)
        # an arbitrary general tolerance
        self.PLUS_MINUS = int(self.MAX_XY * 0.005)
        self.SQUARE_LENGTH = self.SQUARE_RATIO + self.EDGE_RATIO
        self.MAX_INDEX = 9  # assumes grid is square 10x10

    def create_grid(self):
        """ 
        ### Create grid
        Function that creates the grid of squares in the image.
        """
        # These are the stop values for the for loops.
        STOP_XY = self.MAX_XY - self.EDGE_RATIO
        STEP = self.SQUARE_LENGTH

        # iterate through the grid by moving in steps of SQUARE_LENGTH
        # until the max x and y values are reached.
        # x and y are the top left points of the squares
        # x_index and y_index are the index of the square in the grid
        for y, x in itertools.product(range(0, STOP_XY, STEP), range(0, STOP_XY, STEP)):

            # get the corresponding index of the square in the grid
            # flipped coordinates to make it (row, column)
            y_index, x_index = Utils.xy_to_index(self, x, y)

            # coordinates of the top left and bottom right points of the square
            top_left = (
                x + (self.EDGE_RATIO),
                y + (self.EDGE_RATIO)
            )
            bottom_right = (
                x + self.SQUARE_RATIO + (self.EDGE_RATIO),
                y + self.SQUARE_RATIO + (self.EDGE_RATIO)
            )

            # create a square object sq
            sq = Square(
                top_left,
                bottom_right,
                (x_index, y_index),
                self.PIN_RATIO,
                self.PLUS_MINUS,
                self.img
            )

            # add the square to the grid list
            self.grid[x_index][y_index] = sq

    ### Find functions ###

    def find_pins(self, contours: list[np.ndarray]):
        """
        ### Find potential pins
        ---------------
        Function that finds the pins and adds them to their bounding squares.
        The pins are found by finding the square structures in the image.

        #### Args:
        * contours: list of contours around non-grayscale (colorful) edges in image
        """

        # Square structures are 4 points (in this case pins),
        # arranged in the shape of a square
        square_structures, pin_list = self.get_square_structures(contours)

        # adds the 4 potential pins structured as a square shape to the
        # square in the grid where the middle of the structure is located
        for square_structure, pins in zip(square_structures, pin_list):

            # get the middle of the structure
            center = find_center_of_points(square_structure)

            # get the index of sq based on the center of the structure
            # flipped coordinates to make it (row, column)
            y_index, x_index = Utils.xy_to_index(self, center[0], center[1])

            # add pins to the appropriate square in the grid
            for pin in pins:
                self.grid[x_index][y_index].add_pin(pin)

    def find_blocks(self, contours: list[np.ndarray]):
        """
        ### Find blocks
        ---------------
        Function that determines which squares are blocks in the grid.
        It does this by finding the potential pins (p_pins) 
        then checking if the pin is in the corners of the square and adding it to the square if it is.

        #### Args:
        * contours: list of contours around non-grayscale (colorful) edges in image

        #### Returns:
        None
        """

        # finds the potential pins (p_pins)
        # and adds them to their bounding squares.
        self.find_pins(contours)

        # checks if the potential pins are in one of the corners of square.
        # adds potential pin as a pin to the square if it is.
        self.process_pins()

        # checks if the square has x or more pins
        # if it does, it is considered a block.
        self.add_blocks()

    ## Helper functions ##
    def process_pins(self):
        """
        checks if the potential pins are in one of the corners of square.
        Adds potential pin as a pin to the square if it is.
        """
        for sq in itertools.chain(*self.grid):
            if len(sq.get_p_pins()) < 4:
                continue

            for p_pin in sq.get_p_pins():
                x, y, w, h = cv.boundingRect(p_pin)

                # checks if top left or bottom right point of pin
                # is inside corner of square within error range
                if sq.is_in_corners_skewed(x, y, w, h):
                    sq.add_pin(p_pin)

    ### Get functions ###

    def get_blocks(self):
        return self.blocks

    def get_contour_centers(self, contours: list[np.ndarray]):
        """
        Function that finds the center point of each contour

        #### Args:
        * contours: list of contours around non-grayscale (colorful) edges in image

        #### Returns:
        * center_to_contour_index: dictionary with center points as keys and their corresponding contour indices as values
        """
        center_to_contour_index = {}
        for i, contour in enumerate(contours):
            center = find_center_of_contour(contour)
            if center is not None:
                center_to_contour_index[center] = i

        # save the indexes bounding the centers of the
        # contours in a list and remove None values
        centers = list(center_to_contour_index.keys())
        centers = [x for x in centers if x != None]

        return center_to_contour_index, centers

    def get_square_structures(self, contours: list[np.ndarray]):
        """
        ### Square structures
        ---------------
        Function that finds the square structures in the image.  
        A square structure is defined as 4 points (in this case potential pins) arranged in the shape of a square.

        #### Args:
        * contours: list of contours around non-grayscale (colorful) edges in image

        #### Returns:
        * square_structures: list of square structures
        * p_pins: list of p_pins
        """
        square_structures = []
        pins = []

        # find the center of each contour
        center_to_contour_index, centers = self.get_contour_centers(contours)

        # Find all combinations of four points
        combinations = list(itertools.combinations(centers, 4))

        # useful values for debuggin
        """
        point0_step = math.comb(len(centers)-1, 3)
        point1_step = math.comb(len(centers)-2, 2)
        point2_step = math.comb(len(centers)-3, 1)"""
        # print("centers:", len(centers), "combinations:", len(combinations))
        # print("point0_step:", point0_step, "point1_step:", point1_step, "point2_step:", point2_step)
        index = 0
        debug_flag = 0
        #step_filter = point0_step

        # iterate through the combinations of points
        for comb in combinations:
            # (previously) missing block @ 6,4 in image 6066
            '''if index == 1171074: #(1179520 - point1_step - (point2_step*45) - 1):
                print("special: ", index)
                debug_flag = True

            else:
                debug_flag = False
            '''
            if is_arranged_as_square(comb, self.img, self.SQUARE_LENGTH, recursion_flag=0, debug=debug_flag):

                # Add the square to the list of
                # combinations if it is arranged as a square
                square_structures.append(list(comb))

                # Find the indices of the contours that form the square
                contour_indices = [center_to_contour_index[point]
                                   for point in comb]
                pins.append([contours[i] for i in contour_indices])

            index += 1

        return square_structures, pins

    ## Add functions ##

    def add_blocks(self):
        for sq in itertools.chain(*self.grid):
            if len(sq.get_pins()) >= 4:
                sq.is_block = True
                self.blocks.append(sq)

    def add_artificial_block(self, index: tuple[int, int], img, sq_img: np.ndarray):
        """
        ### Add artificial block
        ---------------
        Function that adds an artificial block to the grid.

        #### Args:
        * index: index of the square in the grid
        * sq_img: image of the square
        """

        # get the square in the grid
        i, j = index
        sq = self.grid[i][j]

        # No need to worry about all the square parameters,
        # After the colage, we can pass it through phase 1 again
        # sq.is_block = True, sq.block_type = ...
        return self.paste_block(img, sq_img, sq.tl, sq.br)

    ### Draw functions ###

    def paste_test_area(self, block):
        """ replaces the pixels at the block's location with the block's test area image """
        
        
        # get the top left and bottom right points of the block
        tl, br = block.tl, block.br

        # get the images
        img = self.img
        test_area_img = block.test_area_img

        # pixel coordinates
        corners = block.calculate_corners_pinbased()
        y_min = corners[0][1][1]
        y_max = corners[2][0][1]
        x_min = corners[0][1][0]
        x_max = corners[2][0][0]

        # paste the image of the block's test area on the grid        
        img[y_min:y_max, x_min:x_max] = test_area_img

        self.img = img

    def paste_block(self, img, sq_img, tl, br):
        " pastes the image of the block with transparent bkg on the grid "

        # paste the image of the square on the grid
        # at the top left and bottom right points of the square
        # tl and br are the top left and bottom right points of the square
        center_pt = self.calculate_center(tl, br)
        # sq_img = cv.resize(sq_img, sq_size)

        img = self.add_transparent_image(img, sq_img, center_pt)

        return img

    def add_transparent_image(self, background, foreground, center_pt):
        "Source: https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image"

        bg_h, bg_w, bg_channels = background.shape
        fg_h, fg_w, fg_channels = foreground.shape

        assert bg_channels == 3, f"background image should have exactly 3 channels (RGB). found:{bg_channels}"
        assert fg_channels == 4, f"foreground image should have exactly 4 channels (RGBA). found:{fg_channels}"

        # center the foreground image on the background image according to the center point
        x_offset = center_pt[0] - fg_w // 2
        y_offset = center_pt[1] - fg_h // 2

        w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
        h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

        if w < 1 or h < 1:
            return

        # clip foreground and background images to the overlapping regions
        bg_x = max(0, x_offset)
        bg_y = max(0, y_offset)
        fg_x = max(0, x_offset * -1)
        fg_y = max(0, y_offset * -1)
        foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
        background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

        # separate alpha and color channels from the foreground image
        foreground_colors = foreground[:, :, :3]
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        # combine the background with the overlay image weighted by alpha
        composite = background_subsection * \
            (1 - alpha_mask) + foreground_colors * alpha_mask

        # overwrite the section of the background image that has been updated
        background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

        return background

    def draw_gridLines(self, img: np.ndarray):
        """
        ### draws grid lines
        ---------------
        Function that draws the grid lines on the image.
        """

        # draw grid lines
        start = 0 + self.EDGE_RATIO
        stop = self.MAX_XY - self.EDGE_RATIO + self.PLUS_MINUS
        step = self.SQUARE_RATIO + self.EDGE_RATIO

        for i, j in itertools.product(range(start, stop, step), repeat=2):
            # vertical lines
            cv.line(img, (i, 0), (i, self.MAX_XY), (0, 255, 0), 2)
            cv.line(img, (i + self.SQUARE_RATIO, 0),
                    (i + self.SQUARE_RATIO, self.MAX_XY), (0, 255, 0), 2)

            # horizontal lines
            cv.line(img, (0, i), (self.MAX_XY, i), (0, 255, 0), 2)
            cv.line(img, (0, i + self.SQUARE_RATIO),
                    (self.MAX_XY, i + self.SQUARE_RATIO), (0, 255, 0), 2)

    def draw_blocks(self, image_copy: np.ndarray, show_pins=False, show_corners=False):
        """
        Function that shows image with pins and corners drawn

        #### Args:
        * image_copy: copy of the original image
        * show_pins: boolean to show pins
        * show_corners: boolean to show corners
        """
        for blk in self.blocks:
            blk.draw_pins(image_copy) if show_pins else None
            blk.draw_corners(image_copy) if show_corners else None
            cv.rectangle(image_copy, blk.tl, blk.br, (0, 0, 255), 3)

    def calculate_center(self, lft_top, rgt_bot):
        """
        ### Calculate center
        ---------------
        Function that calculates the center of a square.

        #### Args:
        * lft_top: top left point of the square
        * rgt_bot: bottom right point of the square

        #### Returns:
        * center: center point of the square
        """
        x = (lft_top[0] + rgt_bot[0]) // 2
        y = (lft_top[1] + rgt_bot[1]) // 2
        return (x, y)

    def get_square(self, index):
        """
        ### Get square
        ---------------
        Function that gets the square at the given index.

        #### Args:
        * index: index of the square in the grid

        #### Returns:
        * square: square at the given index
        """
        return self.grid[index[0]][index[1]]

    def set_square(self, index, square):
        """
        ### Set square
        ---------------
        Function that sets the square at the given index.

        #### Args:
        * index: index of the square in the grid
        * square: square to set
        """
        self.grid[index[0]][index[1]] = square

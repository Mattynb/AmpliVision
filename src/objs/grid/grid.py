import cv2 as cv
from matplotlib.pyplot import step
import numpy as np
from .igrid import IGrid
from .Square import Square
import itertools
from ..utils.utils_geometry import is_arranged_as_square, find_center_of_points, find_center_of_contour
from ..utils import Utils
import math

class Grid(IGrid):
    def __init__(self, img: np.ndarray):
        # scanned image
        self.img = img.copy()

        # setup ratios used in the grid 
        # such as the plus minus, etc.
        self.setup_ratios()

        # represents the grid in the image as a 2D array of squares
        self.grid = [[None for _ in range(self.MAX_INDEX + 1)] for _ in range(self.MAX_INDEX + 1)]
        self.blocks = []
        self.create_grid()

    ## Setup functions ##
    def setup_ratios(self):
        """
        ### Setup rations
        Function that sets up the ratios used in the grid.
        """
        # max x and y coordinates of the image
        self.MAX_XY = self.img.shape[0] # assumes image is square

        # ratios measured experimentally as a percentage of the grid
        self.PIN_RATIO = int(self.MAX_XY * 0.012)     # size of pin diameter/grid size
        self.EDGE_RATIO = int(self.MAX_XY * 0.01)     # size of edge/grid size. Edge is the "lines" around squares 
        self.SQUARE_RATIO = int(self.MAX_XY * 0.089)  # squares are the places where you can insert the ampli blocks
        self.PLUS_MINUS = int(self.MAX_XY * 0.005)    # an arbitrary general tolerance
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
            y_index, x_index = Utils.xy_to_index(self, x, y) # flipped coordinates to make it (row, column)

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
        square_structures, pin_list  = self.get_square_structures(contours)

        # adds the 4 potential pins structured as a square shape to the 
        # square in the grid where the middle of the structure is located
        for square_structure, pins in zip(square_structures, pin_list):

            # get the middle of the structure 
            center = find_center_of_points(square_structure)
            
            # get the index of sq based on the center of the structure
            y_index, x_index = Utils.xy_to_index(self, center[0], center[1]) # flipped coordinates to make it (row, column)

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
        point0_step = math.comb(len(centers)-1, 3)
        point1_step = math.comb(len(centers)-2, 2)  
        point2_step = math.comb(len(centers)-3, 1)  
        #print("centers:", len(centers), "combinations:", len(combinations))
        #print("point0_step:", point0_step, "point1_step:", point1_step, "point2_step:", point2_step)
        index = 0; debug_flag = 0; step_filter = point0_step

        # iterate through the combinations of points
        for comb in combinations:
            # (previously) missing block @ 6,4 in image 6066
            if index == 1171074: #(1179520 - point1_step - (point2_step*45) - 1):
                print("special: ", index)
                debug_flag = True

            else:
                debug_flag = False

            if is_arranged_as_square(comb, self.img, self.SQUARE_LENGTH, recursion_flag=0, debug=debug_flag): 

                # Add the square to the list of 
                # combinations if it is arranged as a square
                square_structures.append(list(comb))

                # Find the indices of the contours that form the square
                contour_indices = [center_to_contour_index[point] for point in comb]
                pins.append([contours[i] for i in contour_indices])

            index += 1

        return square_structures, pins


    ## Add functions ##
    def add_blocks(self):
        for sq in itertools.chain(*self.grid):
            if len(sq.get_pins()) >= 4:
                sq.is_block = True
                self.blocks.append(sq)
    

    ### Draw functions ### 
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
            cv.line(img, (i + self.SQUARE_RATIO, 0), (i + self.SQUARE_RATIO, self.MAX_XY), (0, 255, 0), 2)

            # horizontal lines
            cv.line(img, (0, i), (self.MAX_XY, i), (0, 255, 0), 2)
            cv.line(img, (0, i + self.SQUARE_RATIO), (self.MAX_XY, i + self.SQUARE_RATIO), (0, 255, 0), 2)

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
                cv.rectangle(image_copy, blk.tl, blk.br, (0,0,255), 3)

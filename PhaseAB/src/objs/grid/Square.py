from ast import Tuple
import cv2 as cv
import numpy as np
from ..utils.utils_color import *


class Square:
    """
    ### Square
    ---------------
    Class that represents a square in the grid_ds.

    #### Args:
    * tl: top left point of the square
    * br: bottom right point of the square
    * index: index of the square in the grid_ds

    #### Attributes:
    * tl: top left point of the square
    * br: bottom right point of the square
    * index: index of the square in the grid_ds
    * block: boolean that indicates if the square is a block
    * pin_count: number of pins in the square

    #### Methods:
    * add_pin: adds a pin to the square
    * draw_pins: draws the pins in the square
    * draw_corners: draws the corners of the square
    * createImg: creates an image of the square, a cutout of the image around the square
    * add_corners: adds the corners of the square to the square object
    * is_in_corners: checks if a point is in the corners of the square
    * which_corner_is_contour_in: finds which corner of square a contour is in
    * get_rgb_avg_of_contour: gets the average RGB of a contour in the image
    * get_pins_rgb: gets the average RGB of the pins in the square

    """

    def __init__(self, tl: int, br: int, index: Tuple, PIN_RATIO: int, PLUS_MINUS: int, img: np.ndarray) -> None:
        # potential pins
        self.p_pins = []

        # pins
        self.pins = []

        # block or not and type of block
        self.is_block = False
        self.block_type = ''

        # RBG values of the pins in the square (tl, tr, bl, br)
        self.rgb_sequence = []

        # coordinates and index in Grid
        self.tl = tl
        self.br = br
        self.index = index

        # image and image of the square for visualization if necessary
        self.img = img.copy()
        if img is not None:
            self.sq_img = self.createImg(img.copy())

        # corners of the square
        self.corners = []
        self.add_corners(PIN_RATIO, PLUS_MINUS)

        self.test_area_img = None

        # rotation of the block. 0 is vertical strip with bkg on bottom. 1 is horizontal strip with bkg on left side. etc.
        self.rotation = 0

        # ratios
        self.PIN_RATIO = PIN_RATIO
        self.PLUS_MINUS = PLUS_MINUS

    ## Get functions ##
    def get_index(self) -> Tuple:
        """ Returns the index of the square """
        return self.index

    def get_p_pins(self) -> list[int]:
        """ Returns the potential pins in the square """
        return self.p_pins

    def get_pins(self) -> list[int]:
        """ Returns the pins in the square """
        return self.pins

    def get_corners(self) -> list[int]:
        """ Returns the corners of the square """
        return self.corners

    def get_img(self) -> np.ndarray:
        """ Returns the image of the square """
        return self.img

    def get_sq_img(self) -> np.ndarray:
        """ Returns the image of the square """
        if self.sq_img is None:
            self.sq_img = self.createImg(self.img)
        return self.sq_img

    def get_test_area_img(self) -> np.ndarray:
        " Returns the image of squares test area (inner square where test strip can be)"
        if self.test_area_img is None:
            self.test_area_img = self.create_test_area_img(self.get_sq_img())
        return self.test_area_img

    def get_block_type(self) -> str:
        """ Returns the block type of the square """
        if self.is_block:
            return self.block_type
        else:
            return "Not a block"

    def get_rgb_sequence(self) -> list[int]:
        """ Returns the RGB sequence of the square """
        return self.rgb_sequence

    def createImg(self, img: np.ndarray) -> np.ndarray:
        """ Creates an image of the square, a cutout of the image around the square"""
        return img[(self.tl[1]-10):(self.br[1]+10), (self.tl[0]-10):(self.br[0]+10)]

    def create_test_area_img(self, sq_img: np.ndarray) -> np.ndarray:
        " Creates an image of the inner test spot"

        sq_img = self.img
        corners = self.calculate_corners_pinbased()

        """
        r = sq_img[a:b, c:d] means that the image r is a cutout of the image sq_img. 
        from the top left corner (a, c) to the bottom right corner (b, d).
        where a, b, c, d are the coordinates of the corners of the square.
        for example,
        a = corners[0][1][1] means that a is the y coordinate of the bottom right corner of the top right corner of the square.
        b = corners[2][0][1] means that b is the y coordinate of the top left corner of the bottom right corner of the square.
        c = corners[0][1][0] means that c is the x coordinate of the bottom right corner of the top right corner of the square.
        d = corners[2][0][0] means that d is the x coordinate of the top left corner of the bottom right corner of the square.
        """

        return sq_img[corners[0][1][1]:corners[2][0][1], corners[0][1][0]:corners[2][0][0]]

    ## Add functions ##
    def add_pin(self, pin: np.ndarray) -> None:
        """ Adds a pin to the square """
        self.pins.append(pin)

    def add_p_pin(self, pin: np.ndarray) -> None:
        """ Adds a potential pin to the square """
        self.p_pins.append(pin)

    def add_corners(self, PIN_RATIO: int, PLUS_MINUS: int, p: int = 3, a: float = 1.8) -> None:
        """ 
        Adds the corners of the square to the square object

        #### Args:
        * PIN_RATIO: ratio of the pin size to the square size
        * PLUS_MINUS: arbitrary tolerance value
        * p: "padding" value. Determines size of the corners.
        * a: skew value. Is the exponential determining how skewed the corners are.
        """

        # top left and bottom right coordinates of the square
        tl_x, tl_y = self.tl
        br_x, br_y = self.br

        # Skewing the corners in relation to the center of the grid to account for perspective.
        # the further away from the center, the more skewed the corners are (exponential).

        # Avoiding division by zero
        SKEW_x, SKEW_y = self.calculate_skew(a)

        # The following four values: top_right, top_left, bottom_right, bottom_left are the corners of the square.
        # Each corner contains its top left and bottom right coordinates.
        # Coordinates are calculated using:
        # top left and bottom right coordinates of the square, arbitrary plus minus value, the padding value and the skew value.

        self.corners = self.calculate_corners(
            tl_x, tl_y, br_x, br_y, PIN_RATIO, PLUS_MINUS, p, SKEW_x, SKEW_y)

    def calculate_skew(self, a: float) -> Tuple:
        """ 
        Calculates the skew originated from cellphone cameras

        |x-4|^a * (x-4)/|x-4|
        """
        if self.index[0] != 4:
            SKEW_x = int(
                (abs(self.index[0] - 4) ** a) * ((self.index[0] - 4) / abs(self.index[0] - 4)))
        else:
            SKEW_x = 0

        # Avoiding division by zero
        if self.index[1] != 4:
            SKEW_y = int(
                (abs(self.index[1] - 4) ** a) * ((self.index[1] - 4) / abs(self.index[1] - 4)))
        else:
            SKEW_y = 0

        return SKEW_x, SKEW_y

    def calculate_corners(self, tl_x: int, tl_y: int, br_x: int, br_y: int, PIN_RATIO: int, PLUS_MINUS: int, p: int, SKEW_x: int, SKEW_y: int) -> list[int]:
        """
        Calculates the corners of the square using magic. The "corners" here refer to the space in the ampli block where the pins are located.
        """
        top_right = (
            (tl_x - (p*PLUS_MINUS) + SKEW_x, tl_y - (p*PLUS_MINUS) + SKEW_y),
            (tl_x + PIN_RATIO + (p*PLUS_MINUS) + SKEW_x,
             tl_y + PIN_RATIO + (p*PLUS_MINUS) + SKEW_y)
        )

        top_left = (
            (br_x - PIN_RATIO - (p*PLUS_MINUS) +
             SKEW_x, tl_y - (p*PLUS_MINUS) + SKEW_y),
            (br_x + (p*PLUS_MINUS) + SKEW_x, tl_y +
             PIN_RATIO + (p*PLUS_MINUS) + SKEW_y)
        )

        bottom_right = (
            (tl_x - (p*PLUS_MINUS) + SKEW_x, br_y -
             PIN_RATIO - (p*PLUS_MINUS) + SKEW_y),
            (tl_x + PIN_RATIO+(p*PLUS_MINUS) +
             SKEW_x, br_y + (p*PLUS_MINUS) + SKEW_y)
        )

        bottom_left = (
            (br_x - PIN_RATIO - (p*PLUS_MINUS) + SKEW_x,
             br_y - PIN_RATIO - (p*PLUS_MINUS) + SKEW_y),
            (br_x + (p*PLUS_MINUS) + SKEW_x, br_y + (p*PLUS_MINUS) + SKEW_y)
        )
        return [top_right, top_left, bottom_right, bottom_left]

    def calculate_corners_pinbased(self) -> list[list[int]]:
        """
        Calculates the corners of the square based on the pins in the square.
        To be used after the pins have been added to the square.
        list[[corner_tl, corner_br], ...] in clockwise order starting from top left.
        """
        corners = []
        # pin is a list of contours
        for pin in self.pins:
            x, y, w, h = cv.boundingRect(pin)

            # add extra padding to the corners
            px, py = self.calculate_skew(0.2)
            px = int(px)
            py = int(py)

            # append top left and bottom right points of the test area
            corners.append([(x-px, y-py), (x+w+px, y+h+py)])

        return self.order_corner_points(corners)

    ## Drawing functions ##

    def draw_p_pins(self, image: np.ndarray) -> None:
        """ Draws the potential pins in the square """
        for pin in self.p_pins:
            cv.drawContours(image, pin, -1, (0, 255, 0), 3)

    def draw_pins(self, image: np.ndarray) -> None:
        """ Draws the pins in the square """
        for pin in self.pins:
            cv.drawContours(image, pin, -1, (0, 255, 0), 3)

    def draw_corners(self, img: np.ndarray) -> None:
        """ Draws the corners of the square """
        for corner in self.corners:
            cv.rectangle(img, corner[0], corner[1], (0, 0, 255), 1)

    def draw_corners_pinbased(self, img: np.ndarray) -> None:
        """ Draws the corners of the square based on the pins in the square
        To be used after the pins have been added to the square."""
        # pin is a list of contours
        for x, y in self.calculate_corners_pinbased():
            cv.rectangle(img, x, y, (0, 0, 255), 1)

    def draw_test_area(self, img: np.ndarray) -> None:
        "Draws the test area of the square"

        corners = self.calculate_corners_pinbased()
        cv.rectangle(img, corners[0][1], corners[2][0], (0, 0, 255), 1)

    ### Boolean functions ###
    def is_in_test_bounds(self, x: int, y: int) -> bool:
        "checks if coordinate is within test bounds (inner square where strip is)"

        pass

    def is_in_corners(self, x: int, y: int) -> bool:
        """ 
        Checks if a point is in the corners of the square. 
        """
        # corn = ["top_left", "top_right", "bottom_left", "bottom_right"]

        i = 0
        for corner in self.corners:
            if x >= corner[0][0] and x <= corner[1][0]:
                if y >= corner[0][1] and y <= corner[1][1]:
                    # print(corn[i], ": ", round(self.get_rgb_avg_of_contour(contour)))
                    return True
            i += 1

        return False

    def is_in_corners_skewed(self, x: int, y: int, w: float, h: float) -> bool:
        """ Checks if a point is in the corners of the square, 
        taking into consideration the skewing that happens."""
        return (self.is_in_corners(x, y) or
                self.is_in_corners(x+int(w), y+int(h)) or
                self.is_in_corners(x-int(w), y-int(h)) or
                self.is_in_corners(x+int(w), y-int(h)) or
                self.is_in_corners(x-int(w), y+int(h)))

    def which_corner_is_contour_in(self, contour: np.ndarray = None, xy=None) -> str:
        """
        Function that finds which corner of square a contour is in.
        """
        corn = ["top_left", "top_right", "bottom_left", "bottom_right"]

        if xy is None:
            x, y = cv.boundingRect(contour)[:2]
        else:
            x, y = xy
            x = int(x)
            y = int(y)

        i = 0
        for corner in self.corners:
            if x >= corner[0][0] and x <= corner[1][0]:
                if y >= corner[0][1] and y <= corner[1][1]:
                    return corn[i]
            i += 1

        # might be unecessary after corner skewing
        i = 0
        for corner in self.corners:
            if x + (2*self.PLUS_MINUS) >= corner[0][0] and x - (2*self.PLUS_MINUS) <= corner[1][0]:
                if y + (2*self.PLUS_MINUS) >= corner[0][1] and y - (2*self.PLUS_MINUS) <= corner[1][1]:
                    return corn[i]
            i += 1

    def order_corner_points(self, corners: list[int]) -> list[int]:
        """
        Orders the corners of the square in a clockwise manner starting from the top-left corner.
        """
        # top right, top left, bottom right, bottom left
        ordered_corners = [None, None, None, None]

        for xy in corners:
            mid_x = (xy[0][0] + xy[1][0]) / 2
            mid_y = (xy[0][1] + xy[1][1]) / 2
            s = self.which_corner_is_contour_in(xy=(mid_x,  mid_y))

            if s == "top_left":
                ordered_corners[0] = xy
            elif s == "top_right":
                ordered_corners[1] = xy
            elif s == "bottom_right":
                ordered_corners[2] = xy
            elif s == "bottom_left":
                ordered_corners[3] = xy

        if None in ordered_corners:
            print("\nError in ordering corners\n")
            return None

        return ordered_corners

    # set functions

    def set_rgb_sequence(self) -> None:
        """
        ### Set rgb sequence
        ---------------
        Function that sets the rgb sequence of the square.

        #### Returns:
        * None
        """

        # get the RGB values of the pins in the square
        pins_rgb, corner_key = get_pins_rgb(self)

        # fixing the order from tr,tl,br,bl to clockwise starting from top-right. This might be the ugliest code I've ever written. But it works!
        set_rgb_sequence_clockwise(self, pins_rgb, corner_key)

    def set_test_area_img(self, img):
        " Sets the image of the test area"
        self.test_area_img = img

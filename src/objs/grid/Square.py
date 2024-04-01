from ast import Tuple
import cv2 as cv
import numpy as np

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
    def __init__(self, tl: int, br: int, index: Tuple, PIN_RATIO: int, PLUS_MINUS: int, img: np.ndarray = None) -> None:
        # potential pins
        self.p_pins = []

        # pins
        self.pins = []

        # block or not and type of block
        self.is_block = False
        self.block_type = ''
        self.rgb_sequence = [] #RBG values of the pins in the square (tl, tr, bl, br)

        # coordinates and index in Grid
        self.tl = tl; self.br = br
        self.index = index

        # image and image of the square for visualization if necessary
        self.img = img.copy()
        self.sq_img = self.createImg(img.copy()) if img is not None else None

        # corners of the square
        self.corners = []
        self.add_corners(PIN_RATIO, PLUS_MINUS)
        
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
    
    def get_block_type(self)->str:
        """ Returns the block type of the square """
        if self.is_block:
            return  self.block_type
        else:
            return "Not a block"

    def get_rgb_sequence(self)->list[int]:
        """ Returns the RGB sequence of the square """
        return self.rgb_sequence
    
    def createImg(self, img: np.ndarray)->np.ndarray:
        """ Creates an image of the square, a cutout of the image around the square"""
        return img[(self.tl[1]-10):(self.br[1]+10), (self.tl[0]-10):(self.br[0]+10)]

    ## Add functions ##
    def add_pin(self, pin: np.ndarray)->None:
        """ Adds a pin to the square """
        self.pins.append(pin)
    
    def add_p_pin(self, pin: np.ndarray)->None:
        """ Adds a potential pin to the square """
        self.p_pins.append(pin)

    def add_corners(self, PIN_RATIO: int, PLUS_MINUS:int, p:int =3, a:float = 1.8) -> None:
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
        if self.index[0] != 4:
            SKEW_x = int((abs(self.index[0] - 4) ** a  ) * ((self.index[0] - 4)/ abs(self.index[0] - 4))) 
        else:
            SKEW_x = 0

        # Avoiding division by zero
        if self.index[1] != 4:
            SKEW_y = int((abs(self.index[1] - 4) ** a  ) * ((self.index[1] - 4)/ abs(self.index[1] - 4))) 
        else:
            SKEW_y = 0


        # The following four values: top_right, top_left, bottom_right, bottom_left are the corners of the square.
        # Each corner contains its top left and bottom right coordinates.
        # Coordinates are calculated using:
        # top left and bottom right coordinates of the square, arbitrary plus minus value, the padding value and the skew value.
       
        self.corners = self.calculate_corners(tl_x, tl_y, br_x, br_y, PIN_RATIO, PLUS_MINUS, p, SKEW_x, SKEW_y)


    def calculate_corners(self, tl_x:int, tl_y:int, br_x:int, br_y:int, PIN_RATIO:int, PLUS_MINUS:int, p:int, SKEW_x:int, SKEW_y:int) -> list[int]:
        """
        Calculates the corners of the square using magic. The "corners" here refer to the space in the ampli block where the pins are located.
        """
        top_right = (
            ( tl_x - (p*PLUS_MINUS) + SKEW_x, tl_y - (p*PLUS_MINUS) + SKEW_y),
            ( tl_x + PIN_RATIO + (p*PLUS_MINUS) + SKEW_x, tl_y + PIN_RATIO + (p*PLUS_MINUS) + SKEW_y)
        )

        top_left = (
            ( br_x - PIN_RATIO - (p*PLUS_MINUS) + SKEW_x, tl_y - (p*PLUS_MINUS) + SKEW_y),
            ( br_x + (p*PLUS_MINUS) + SKEW_x, tl_y + PIN_RATIO + (p*PLUS_MINUS) + SKEW_y)
        )

        bottom_right = (
            ( tl_x - (p*PLUS_MINUS) + SKEW_x, br_y - PIN_RATIO - (p*PLUS_MINUS) + SKEW_y),
            ( tl_x + PIN_RATIO+(p*PLUS_MINUS) + SKEW_x, br_y + (p*PLUS_MINUS) + SKEW_y)
        )

        bottom_left = (
            ( br_x - PIN_RATIO - (p*PLUS_MINUS) + SKEW_x, br_y - PIN_RATIO - (p*PLUS_MINUS) + SKEW_y),
            ( br_x + (p*PLUS_MINUS) + SKEW_x, br_y + (p*PLUS_MINUS) + SKEW_y)
        )
        return [top_right, top_left, bottom_right, bottom_left]
    

    ## Drawing functions ##
    def draw_pins(self, image: np.ndarray) -> None:
        """ Draws the pins in the square """
        for pin in self.pins:
            cv.drawContours(image, pin, -1, (0, 255, 0), 1)      
    
    def draw_corners(self, img: np.ndarray) -> None:
        """ Draws the corners of the square """
        for corner in self.corners:
            cv.rectangle(img, corner[0], corner[1], (0, 0, 255), 1)


    ### Boolean functions ###
    def is_in_test_bounds(self, x:int, y:int) -> bool:
        "checks if coordinate is within test bounds (inner square where strip is)"

        pass

    def is_in_corners(self, x:int, y:int) -> bool:
        """ 
        Checks if a point is in the corners of the square. 
        """
        #corn = ["top_left", "top_right", "bottom_left", "bottom_right"]

        i = 0
        for corner in self.corners:
            if x >= corner[0][0] and x <= corner[1][0]:
                if y >= corner[0][1] and y <= corner[1][1]:
                    #print(corn[i], ": ", round(self.get_rgb_avg_of_contour(contour)))
                    return True
            i += 1

        return False  
    
    def is_in_corners_skewed(self, x:int, y:int, w:float, h:float) -> bool:
        """ Checks if a point is in the corners of the square, 
        taking into consideration the skewing that happens."""
        return (self.is_in_corners(x, y) or
            self.is_in_corners(x+int(w), y+int(h)) or
            self.is_in_corners(x-int(w), y-int(h)) or
            self.is_in_corners(x+int(w), y-int(h)) or
            self.is_in_corners(x-int(w), y+int(h)))

    def which_corner_is_contour_in(self, contour:np.ndarray) -> str:
        """
        Function that finds which corner of square a contour is in.
        """
        corn = ["top_left", "top_right", "bottom_left", "bottom_right" ]

        x, y = cv.boundingRect(contour)[:2]

        i = 0
        for corner in self.corners:
            if x >= corner[0][0] and x <= corner[1][0]:
                if y >= corner[0][1] and y <= corner[1][1]:
                    return corn[i]
            i += 1

        # might be unecessary after corner skewing
        i =0
        for corner in self.corners:
            if x + (2*self.PLUS_MINUS) >= corner[0][0] and x - (2*self.PLUS_MINUS) <= corner[1][0]:
                if y + (2*self.PLUS_MINUS)  >= corner[0][1] and y - (2*self.PLUS_MINUS)  <= corner[1][1]:
                    return corn[i]
            i += 1

    ## RGB get functions ##
    def get_rgb_avg_of_contour(self, contour:np.ndarray, corner:list[np.ndarray]='') -> list[int]:
        """
        ### Get RGB average of contour
        ---------------
        Function that gets the average RGB of a contour in the image.
        
        #### Args:
        * contour: Contour of the object in the image.
        * corner: Corner of the square the contour is in.

        #### Returns:
        * avg_color: Average RGB color of the contour.
        """

        # copy the image
        image = self.img.copy()

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB format

        (x, y), radius = cv.minEnclosingCircle(contour)

        center = (int(x), int(y))
        radius = int(radius) - int(self.PLUS_MINUS/2.5)

        # get the pixels inside the minEnclosingCircle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv.circle(mask, center, radius, (255), -1)
        pixels_inside = image[mask == 255]

        # Calculate the average RGB values
        average_rgb = np.mean(pixels_inside, axis=0)

        # Remove NaN values
        average_rgb = np.nan_to_num(average_rgb)

        return [round(x) for x in average_rgb ]

    def get_pins_rgb(self) -> tuple[list[int], list[int]]:
        """ 
        gets the average RGB of the pins in the square.
        """

        pins_rgb = []
        corner   = []

        # for each pin in the square get the average RGB value of the pin and its corner
        for pins in self.pins:
            corner.append(self.which_corner_is_contour_in(pins))
            pins_rgb.append(self.get_rgb_avg_of_contour(pins, corner))

        return pins_rgb, corner  # tr, tl, br, bl corners 

    def set_rgb_sequence(self)->None:
        """
        ### Set rgb sequence
        ---------------
        Function that sets the rgb sequence of the square.
        
        #### Returns:
        * None
        """

        # get the RGB values of the pins in the square
        pins_rgb, corner_key = self.get_pins_rgb()
        
        # fixing the order from tr,tl,br,bl to clockwise starting from top-right. This might be the ugliest code I've ever written. But it works!
        self.set_rgb_sequence_clockwise(pins_rgb, corner_key)
    
    def set_rgb_sequence_clockwise(self, pins_rgb: list[int], corner_key: list[int])->None:
        """sets the rgb sequence of the square in clockwise order starting from top-left."""
        
        sequence = []
        for key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
            sequence.append(pins_rgb[corner_key.index(key)] if key in corner_key else (0,0,0))
        
        self.rgb_sequence = sequence
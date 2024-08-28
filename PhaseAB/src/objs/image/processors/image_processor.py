import numpy as np
import cv2 as cv
from ..utils.image_white_balancer import WhiteBalanceAdjuster

class ColorContourExtractor:
    """"   
    ## ColorContourExtractor
    
    This class is responsible for processing an image to isolate the color of the pins.
    
    ### Methods
    - `process_image(scanned_image: np.ndarray) -> np.ndarray`
        - This method pre-processes the image to isolate the color of the pins.
        
    - `show_result(edges: np.ndarray) -> None`
        - This method shows the result of the pre-processing.
    
    ### Example
    ```python
    import cv2 as cv
    import numpy as np
    from src.objs.image.processors.image_processor import ImageProcessor

    scanned_image = cv.imread('path/to/image.jpg')
    edges = ImageProcessor.process_image(scanned_image)
    ImageProcessor.show_result(edges)
    ```
    """

    # A function that pre-processes the image to isolate the color of the pins.
    @staticmethod
    def process_image(
        scanned_image: np.ndarray, 
        hsv_lower = [0, 55, 0], 
        hsv_upper = [360, 255,255], 
        double_thresh:bool = False, 
        display:bool=False) -> np.ndarray:
        """ this method pre-processes the image to isolate the color of the pins."""

        # Copy the image to avoid modifying the original image
        scanned_image_copy = scanned_image.copy()
        
        # Convert the image to HSV color space. Hue Saturation Value. 
        # Similar to RGB but more useful for color isolation.
        img_hsv = cv.cvtColor(scanned_image_copy, cv.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the color you want to isolate
        # These values are the product of trial and error and are not necessarily perfect.
        hsv_lower_color = np.array(hsv_lower)
        hsv_upper_color = np.array(hsv_upper)

        # Create a mask to filter out the grayscale colors isolating the color of the pins.
        color_mask = cv.inRange(img_hsv, hsv_lower_color, hsv_upper_color)

        # Visualize the mask on top of the original image before thresholding
        mask_before_thresholding = color_mask.copy()

        edges = cv.Canny(color_mask, 0, 255)

        if double_thresh:
            
            second_mask = cv.bitwise_and(scanned_image_copy, scanned_image_copy, mask=color_mask)
            #cv.imshow('bitwise and image + mask 1', cv.resize(color_mask,(200,200)))
            
            # make all black pixels white
            second_mask[second_mask == 0] = 255
            #cv.imshow('make black pixels white',  cv.resize(color_mask, (200, 200)))
            
            #  thresholding
            second_mask = cv.cvtColor(second_mask, cv.COLOR_BGR2GRAY)
            #cv.imshow('grey',  cv.resize(color_mask, (200, 200)))
            
            second_mask = cv.bitwise_not(second_mask) 
            #cv.imshow('bitwise not',  cv.resize(second_mask, (200, 200)))
            #cv.waitKey(0)

            cv.threshold(second_mask, 125, 255, cv.THRESH_BINARY, second_mask)
            
            edges = cv.Canny(second_mask, 0, 255)
            
            #cv.imshow('second thresholding ',  cv.resize(
             #   cv.bitwise_and(scanned_image_copy,scanned_image_copy, mask=second_mask), (400, 400)))   
            #cv.imshow('first thresholding ',  cv.resize(
              #  cv.bitwise_and(scanned_image_copy,scanned_image_copy, mask=color_mask), (400, 400)))
            #cv.waitKey(0)

            
        #cv.destroyAllWindows()
            
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
        if display:
            ColorContourExtractor.show_result(contours, scanned_image_copy)

        return contours
    
    # Show the result of the pre-processing.
    @staticmethod
    def show_result(contours: np.ndarray, image) -> None:
        """ this method shows the result of the pre-processing."""

        copy = image.copy()
        cv.drawContours(copy, contours, -1, (0, 255, 0), 1)
        copy = cv.resize(copy, (400, 400))
        cv.imshow('Color Contour Extractor', copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
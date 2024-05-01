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
    def process_image(scanned_image: np.ndarray, hsv_lower = [0, 55, 0], hsv_upper = [180, 255,255], display:bool=False) -> np.ndarray:
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
        edges = cv.Canny(color_mask, 0, 255)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Apply white balance to the original image to make the color of the pins more accurate.
        
        'TODO: This is literally doing nothing here. The scanned image is not returned.'
        #scanned_image = WhiteBalanceAdjuster.adjust(scanned_image)

        if display:
            ColorContourExtractor.show_result(contours, scanned_image)

        return contours
    
    # Show the result of the pre-processing.
    @staticmethod
    def show_result(contours: np.ndarray, image) -> None:
        """ this method shows the result of the pre-processing."""

        copy = image.copy()
        cv.drawContours(copy, contours, -1, (0, 255, 0), 1)
        cv.imshow('Color Contour Extractor', copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        
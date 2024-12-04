import cv2 as cv
import numpy as np


class ContourFinder:
    """
    ## ContourFinder
    This class is responsible for finding contours of the grid in an image.

    ### Methods
    - `find_contours(image: np.ndarray) -> list`
        - This method finds the contours of the grid in the given image and returns the top 5 contours sorted by area.

    ### Example
    ```python
    import cv2 as cv
    from src.objs.image.detectors.contour_finder import ContourFinder

    image = cv.imread('path/to/image.jpg')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 50, 150)
    contours = ContourFinder.find_contours(edged)
    ```

    ## reference
    https://learnopencv.com/automatic-document-scanner-using-opencv/
    """
    @staticmethod
    def find_contours(image: np.ndarray) -> list:
        """ This method finds the contours of the Grid in the given image and returns the top 5 contours sorted by area."""

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(image, (11, 11), 0)

        # EDGE DETECTION
        canny = cv.Canny(blurred, 0, 200)
        canny = cv.dilate(canny, cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (5, 5)))

        # CONTOUR DETECTION
        contours, _ = cv.findContours(
            canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        return sorted(contours, key=cv.contourArea, reverse=True)[:5]

import cv2 as cv
import numpy as np

class CornerDetector:
    """
    ## CornerDetector
    
    The `CornerDetector` class is responsible for detecting the corners of a given image given the contours.

    ### Methods
    - `detect_corners(contours: list, img: np.ndarray) -> list`
        - This method detects the corners of a given image given the contours and returns the corners.

    ## References
    https://learnopencv.com/automatic-document-scanner-using-opencv/
    """
    @staticmethod
    def detect_corners(contours: list, img:np.ndarray)->list:
        """
        This method detects the corners of a given image given the contours and returns the corners.
        Currently being used for the grid detection.
        """

        # Loop over the contours.
        for c in contours:
            # Approximate the contour.
            epsilon = 0.02 * cv.arcLength(c, True)
            corners = cv.approxPolyDP(c, epsilon, True)
            # If our approximated contour has four points
            if len(corners) == 4:
                break
    
        # Sorting the corners and converting them to desired shape.
        corners = sorted(np.concatenate(corners).tolist())

        return CornerDetector.order_points(corners)
    
    @staticmethod
    def order_points(pts: list)->list:
        # Initialising a list of coordinates that will be ordered.
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)

        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]

        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]

        # Computing the difference between the points.
        diff = np.diff(pts, axis=1)

        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        
        # Return the ordered coordinates.
        return rect.astype('int').tolist()
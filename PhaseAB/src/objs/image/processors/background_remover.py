import cv2 as cv
import numpy as np


class BackgroundRemover():
    """
    ## BackgroundRemover
    
    This class is used to remove the background from an image, isolating the foreground object (such as a document).
    
    ### Methods
    - `remove_background(image: np.ndarray) -> np.ndarray`
        - This method removes the background from the given image and returns the image with the background removed.
    
    ### Example
    ```python
    import cv2 as cv
    from src.objs.image.processors.background_remover import BackgroundRemover
    
    image = cv.imread('path/to/image.jpg')
    processed_image = BackgroundRemover.remove_background(image)
    ```

    ## reference
    https://learnopencv.com/automatic-document-scanner-using-opencv/
    """
    @staticmethod
    def remove_background(image: np.ndarray) -> np.ndarray:
        """This method removes the background from the given image and returns the image with the background removed."""
        
        # Create a mask and initialize the background and foreground model
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Define the rectangle for the object
        rect = (20, 20, image.shape[1]-20, image.shape[0]-20)

        # Apply grabCut algorithm to remove the background
        cv.grabCut(image, mask, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT)

        # If mask is 3 or 1, change it to 1 or 0, because we want the background to be black and the foreground to be white
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Multiply the image with the mask to remove the background
        im = image * mask2[:, :, np.newaxis]

        return im
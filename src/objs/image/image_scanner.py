import dis
import cv2 as cv
import numpy as np

from .detectors.contour_finder import ContourFinder
from .detectors.corner_detector import CornerDetector

from .processors.morphological_transformer import MorphologicalTransformer
from .processors.background_remover import BackgroundRemover

def display(image, t=100):
    im = cv.resize(image, (800,800))
    cv.imshow('ImageScanner', im)
    cv.waitKey(t)
    cv.destroyAllWindows()

class ImageScanner:
        """
        Class to scan the image and return the scanned image.
        
        ## Methods:
        - `scan(image_og: np.ndarray) -> np.ndarray`
            - This method scans the image and returns the scanned image.    
        
        - `morphological_transform(gpu_img: cv.cuda_GpuMat) -> cv.cuda_GpuMat`
                - This method applies morphological transformations to highlight the grid.
        
        - `remove_background(img: np.ndarray) -> np.ndarray`
                - This method gets rid of the background through masking + grabcut algorithm.
        
        - `find_contours(gpu_img: cv.cuda_GpuMat) -> list`
                - This method finds the contours of the image.
        
        - `detect_corners(contours: list, img: np.ndarray) -> list`
                - This method detects the corners of the grid.
        
        - `perspective_transform(img: np.ndarray, corners: list) -> np.ndarray`
                - This method applies perspective transform to the image.
        
        - `find_dest(pts: list) -> list`
                - This method finds the destination coordinates.
        
        - `order_points(pts: list) -> list`
                - This method orders the points.
        
        ## reference
            https://learnopencv.com/automatic-document-scanner-using-opencv/
        """

        @classmethod
        def scan(cls, img: np.ndarray)->np.ndarray:
                # Applying morphological transformations to highlight the grid
                # Utilizing the GPU for faster processing
                morph_img = MorphologicalTransformer.apply_morph(img)

                #display(morph_img.copy(), 0)

                # Isolate the grid by removing background (Only works with CPU)
                no_bkg_img = BackgroundRemover.remove_background(morph_img)
                
                #display(no_bkg_img.copy(), 0)

                # Adjusting the image to highlight the grid
                contours = ContourFinder.find_contours(no_bkg_img)
                
                #a = no_bkg_img.copy()
                #cv.drawContours(a, contours, -1, (0, 255, 0), 3)
                #display(a, 0)
                corners = CornerDetector.detect_corners(contours, no_bkg_img)

                final_image = cls.perspective_transform(img, corners)
                
                return final_image
        
        # should go to an util class
        @classmethod
        def transfer_to_gpu(cls, gpu_image: cv.cuda_GpuMat, image: np.ndarray = None, to_gray=False, to_bgr=False) -> cv.cuda_GpuMat:
                if image is not None:
                        gpu_image.upload(image)
                
                if to_gray:
                        gpu_image = cv.cuda.cvtColor(gpu_image, cv.COLOR_BGR2GRAY)
                elif to_bgr:
                        gpu_image = cv.cuda.cvtColor(gpu_image, cv.COLOR_GRAY2BGR)

                return gpu_image

        # should go to an util class        
        @classmethod
        def transfer_to_cpu(cls, gpu_image: cv.cuda_GpuMat, to_gray=False, to_bgr=False) -> np.ndarray:
                if to_gray:
                        gpu_image = cv.cuda.cvtColor(gpu_image, cv.COLOR_BGR2GRAY)
                elif to_bgr:
                        gpu_image = cv.cuda.cvtColor(gpu_image, cv.COLOR_GRAY2BGR)
                
                return gpu_image.download()

        @classmethod
        def perspective_transform(cls, img: np.ndarray, corners: list)->np.ndarray:
                # REARRANGING THE CORNERS 
                destination_corners = cls.find_dest(corners)
                
                # Getting the homography. (aka scanning the image)
                M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
                
                # Perspective transform using homography.
                final = cv.warpPerspective(img, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv.INTER_LINEAR)
        
                return final

        @classmethod
        def find_dest(cls, pts: list)->list:
                # DESTINATION COORDINATES
                (tl, tr, br, bl) = pts

                # Finding the maximum width.
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                # Finding the maximum height.
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                # Final destination co-ordinates.
                destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
                return cls.order_points(destination_corners)

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
                
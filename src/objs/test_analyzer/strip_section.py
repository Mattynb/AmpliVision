from calendar import c
import cv2 as cv
import numpy as np
from ..utils.utils_color import get_rgb_avg_of_contour


class StripSection:
    "This class is responsible for holding data and processes regarding sections of test inner square (bkg, test, or control)"

    def __init__(self, test_square_img:np.ndarray, strip_type: str):
        self.stip_type = strip_type
        self.bounds = self.set_bounds(test_square_img)
        self.spots = [] # each spot is hashmap {"contour": np.ndarray, "avg_rgb": int, "positive": bool}
        self.total_avg_rgb = None

    def add_spot(self, block, contour:np.ndarray, result: bool) -> None:
        " adds spot to section as a hashmap with \"color\" and \"avg_rgb\" "

        avg_rgb = get_rgb_avg_of_contour(block, contour)

        self.spots.append({
            "contour" : contour, 
            "avg_rgb" : list(avg_rgb),
            "positive" : result
        })

        self.set_total_avg_rgb()

    def set_spots_manually(self, block)->None:
        "mostly used to find negative result spots using ratios" 

        copy = block.get_test_area_img().copy()
    
        val = self.bounds
        spot = self.identify_spot_manually(copy, (int((val[0] + val[2])/2), int((val[1] + val[3])/2)), False)
        cv.rectangle(copy, (val[0], val[1]), (val[2], val[3]), (0, 255, 0), 1)
        
        cv.imshow('set_spots_manually()', copy) 
        cv.waitKey(0)
        cv.destroyAllWindows()


        self.add_spot(block, spot, False)

    def identify_spot_manually(self, test_area_img, circ_center, b: bool) -> np.ndarray:
        "used to identify a spot manually"

        # get the contour of circle spot
        copy = test_area_img.copy()
        cv.circle(copy, (circ_center[0], circ_center[1]), 5, (0, 0, 255), 1)
        
        # isolate the red circle color
        hsv = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
        lower_red = np.array([0, 250, 250])
        upper_red = np.array([0, 255, 255])

        mask = cv.inRange(hsv, lower_red, upper_red)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        copy = cv.drawContours(copy, contours, -1, (255, 0, 0), 3)

        return contours[0]

    def set_total_avg_rgb(self, bkg = [0, 0, 0]) -> list[int]:
        "sets the total avg rgb by adding the spot rgb avgs together" 

        if len(self.spots) == 0:
            print("please add spots to section before calling set_total_avg_rgb()")
            return None

        i = 0  
        total_avg = [0, 0, 0]

        # adding the total avg with each spot avg
        for spot in self.spots:
            total_avg = list(map(lambda total, spot_avg: total + spot_avg, total_avg, spot["avg_rgb"]))
            i += 1

        # dividing by the number of spots
        total_avg = list(map(lambda total: total/i, total_avg))

        self.total_avg_rgb = total_avg

    # geometry 
    def set_bounds(self, test_square_img: np.ndarray, orientation = None) -> list[int]:
    
        # image bounds
        x, y, = 0, 0
        h, w = test_square_img.shape[:2] # Shape returns: (height, width, channels)
        bounds = None
        
        # divide the square into 3 sections along the middle strip (bkg, test, control)
        
        # ASSUMPTION:this order ASSUMES the strip is vertical with bkg on bottom
        
        if self.stip_type == "bkg":
            bounds = [x+int(w/3), y+int(3/4*h), int(2/3*w), h]

        elif self.stip_type == "test":
            bounds = [x+int(w/3), y+int(h/4+h/12), int(2/3*w), int(3/4*h)]
        
        elif self.stip_type == "control":
            bounds = [x+int(w/3), y, int(2/3*w), int(h/4+h/12)]   

        return bounds

    def bounds_contour(self, contour) -> bool:
        "checks if contour is within section bounds"
        
        x, y, w, h = cv.boundingRect(contour)
    
        if x > self.bounds[0] and y > self.bounds[1] and x+w < self.bounds[2] and y+h < self.bounds[3]:
            return True
        return False
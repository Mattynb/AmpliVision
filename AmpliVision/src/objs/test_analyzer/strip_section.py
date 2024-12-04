import cv2 as cv
import numpy as np
from ..utils.utils_color import get_rgb_avg_of_contour
import matplotlib.pyplot as plt

class StripSection:
    "This class is responsible for holding data and processes regarding sections of test inner square (bkg, test, or control)"

    def __init__(self, test_square_img: np.ndarray, strip_type: str, rotation: int):
        self.strip_type = strip_type
        self.bounds = self.set_bounds(test_square_img, rotation)
        # each spot is hashmap {"contour": np.ndarray, "avg_rgb": int, "positive": bool}
        self.spots = []
        self.total_avg_rgb = None

    def __str__(self):
        return f"StripSection of type {self.strip_type} with bounds {self.bounds} and {len(self.spots)} spots."

    def print_spots(self):
        "prints the spots in the section"

        for spot in self.spots:
            print(f"{self.strip_type} spot: ",
                  spot["avg_rgb"], " positive: ", spot["positive"])

    def add_spot(self, block, contour: np.ndarray, result: bool, debug: bool = False) -> None:
        " adds spot to section as a hashmap with \"color\" and \"avg_rgb\" "
        
        avg_rgb = get_rgb_avg_of_contour(block, contour)

        print(f"spot: {avg_rgb} positive: {result}") if debug else None
    
        if result == False:
            avg_rgb = (0,0,0)

        self.spots.append({
            "contour": contour,
            "avg_rgb": list(avg_rgb),
            "positive": result
        })

        if debug:
            copy = block.get_test_area_img().copy()
            val = self.bounds
            cv.rectangle(copy, (val[0], val[1]),
                         (val[2], val[3]), (0, 255, 0), 1)
            cv.drawContours(copy, [contour], -1, (0, 0, 255), 1)

            copy = cv.resize(copy, (200, 200))
            """
            cv.imshow('stripSection/add_spot()', copy)
            cv.waitKey(0)#"""
            cv.destroyAllWindows()#"""

        self.set_total_avg_rgb()

    def set_spots_manually(self, block, debug: bool = False) -> None:
        "mostly used to find negative result spots using ratios"

        copy = block.get_test_area_img().copy()

        val = self.bounds
        spot_center = (
            int((val[0] + val[2])/2),
            int((val[1] + val[3])/2)
        )

        spot = self.identify_spot_manually(
            copy,
            spot_center,
            debug=debug
        )

        # result should be set to True on real cases where spot is hard to find with hsv
        # otherwise, set to False
        self.add_spot(block, spot, True, debug=debug)

    def identify_spot_manually(self, test_area_img, circ_center, debug: bool) -> np.ndarray:
        "used to identify a spot manually"

        # get the contour of circle spott
        copy = test_area_img.copy()
        cv.circle(copy, (circ_center[0], circ_center[1]), 5, (0, 0, 255), 1)

        # isolate the red circle color
        hsv = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
        lower_red = np.array([0, 250, 250])
        upper_red = np.array([0, 255, 255])

        mask = cv.inRange(hsv, lower_red, upper_red)
        contours, _ = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if debug:
            copy = cv.drawContours(copy, contours, -1, (255, 0, 0), 1)
            copy = cv.resize(copy, (200, 200))
            plt.imshow(copy)
            plt.show()
    
        return contours[0]

    def set_total_avg_rgb(self, bkg=[0, 0, 0]) -> list[int]:
        "sets the total avg rgb by adding the spot rgb avgs together"

        #assert len(self.spots) != 0, """
        #please add spots to section before calling set_total_avg_rgb()"""
        if len(self.spots) == 0:
            self.total_avg_rgb = 0
            return

        i = 0
        total_avg = [0, 0, 0]

        # adding the total avg with each spot avg
        for spot in self.spots:
            total_avg = list(
                map(
                    lambda total, spot_avg: 
                    total + spot_avg
                    ,total_avg, spot["avg_rgb"]  
                )
            )
            i += 1

        # dividing by the number of spots
        total_avg = list(
            map(
                lambda total: 
                total/i
                , total_avg
            )
        )

        self.total_avg_rgb = total_avg

    def subtract_bkg(self, bkg_rgb_avg: list[int]) -> list[int]:
        "subtracts the bkg rgb from the total avg rgb"

        if self.total_avg_rgb == None:
            print("please set the total avg rgb before calling subtract_bkg()")
            return None

        return list(
            map(
                lambda total, bkg: 
                bkg - total
                , self.total_avg_rgb, bkg_rgb_avg
            )
        )

    # geometry
    def set_bounds(self, test_square_img: np.ndarray, rotation: int) -> list[int]:
        # test strip component bounds
        x, y, = 0, 0
        # Shape returns: (height, width, channels)
        h, w = test_square_img.shape[:2]
        bounds = None

        # divide the square into 3 sections along the middle strip (bkg, test, control)

        # ASSUMPTION: this order ASSUMES the strip is vertical with bkg on bottom

        if rotation == 0:
            if self.strip_type == "spot2":
                # [top left x, top left y, width, height]
                bounds = [x+int(w/3), y, x+int(2/3*w), y+int(h/4)]

            elif self.strip_type == "spot1":
                bounds = [x+int(w/3), y+int(h/3), x+int(2/3*w), y+int(2/3*h)]

            if self.strip_type == "bkg":
                bounds = [x+int(w/3), y+int(3/4*h), x+int(2/3*w), y+h]

        elif rotation == 90:
            if self.strip_type == "bkg":
                bounds = [x, y+int(h/3), x+int(w*1/4), y+int(2/3*h)]

            elif self.strip_type == "spot1":
                bounds = [x+int(w/3), y+int(h/3), x+int(2/3*w), y+int(2/3*h)]

            elif self.strip_type == "spot2":
                bounds = [x+int(w*3/4), y+int(h/3), x+w, y+int(h*2/3)]

        elif rotation == 180:
            if self.strip_type == "bkg":
                # [top left x, top left y, width, height]
                bounds = [x+int(w/3), y, x+int(2/3*w), y+int(h/4)]

            elif self.strip_type == "spot1":
                bounds = [x+int(w/3), y+int(h/3), x+int(2/3*w), y+int(2/3*h)]

            if self.strip_type == "spot2":
                bounds = [x+int(w/3), y+int(3/4*h), x+int(2/3*w), y+h]

        elif rotation == 270:
            if self.strip_type == "spot2":
                bounds = [x, y+int(h/3), x+int(w*1/4), y+int(2/3*h)]

            elif self.strip_type == "spot1":
                bounds = [x+int(w/3), y+int(h/3), x+int(2/3*w), y+int(2/3*h)]

            elif self.strip_type == "bkg":
                bounds = [x+int(w*3/4), y+int(h/3), x+w, y+int(h*2/3)]

        return bounds

    def bounds_contour(self, contour) -> bool:
        "checks if contour is within section bounds"

        x, y, w, h = cv.boundingRect(contour)

        center_point = (x+w/2, y+h/2)

        if center_point[0] >= self.bounds[0] and center_point[0] <= self.bounds[2] and center_point[1] >= self.bounds[1] and center_point[1] <= self.bounds[3]:
            return True

        return False

    def paint_spot(self, block_img, rgb, display=False):
        "paints the spot on the block image"

        if len(self.spots) > 0:
            block_img = cv.drawContours(block_img, [self.spots[-1]["contour"]], -1, rgb, -1)

        if display:
            cv.imshow('paint_spot()', block_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return block_img

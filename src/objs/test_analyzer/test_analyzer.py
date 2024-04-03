import csv
import numpy as np
from datetime import datetime
from ..utils.utils_color import get_rgb_avg_of_contour
from ..image.processors.image_processor import ColorContourExtractor

class TestAnalyzer:
    "This class is responsible for getting and analyzing test results a.k.a phase B"

    def __init__(self, test_square_img: np.ndarray, grid_index: int, block_type: str):

        # look only at the inner test square:
        test_square_img = test_square_img #square.get_test_square()

        # square used in csv export
        self.grid_index = grid_index
        self.block_type = block_type

        strip_sections = {
            "bkg" : StripSection(test_square_img, 'bkg'), 
            "test" : StripSection(test_square_img, 'test'),
            "control" : StripSection(test_square_img, 'control')
        }

    def analyze_test_result(self): # should I name it main?
        "gets test results from a block, analyses them, and export them to csv"
        
        # find the positive spots with hsv mask
        # need to think about cases where mask for example return one pixel. 
        #   do you check for minimum contour size? do you only look for it manually? food for thought 
        rgb_spots = ColorContourExtractor.process(self.test_sq_img, lower_hsv= [...], upper_hsv= [...])
        self.add_positives_to_sections(rgb_spots)

        # get background color noise so we can remove it from other sections
        self.strip_sections['bkg'].set_total_avg_rgb()
        bkg_rgb_avg = self.strip_sections['bkg'].total_avg_rgb

        # find the negative spots "manually" through ratios, removing bgk
        self.add_negatives_to_sections(bkg_rgb_avg)

        # validate results to catch any potential errors in the test
        self.validate_results()

        # export results to csv
        self.export_to_csv()

    def add_positives_to_sections(self, rgb_spots) -> None:
        "used to add positive result spots to appropriate strip section"

        # adds each spot to its strip section
        for spot in rgb_spots:
            for section in self.strip_sections.values():
                if section.bounds_contour(spot):
                    section.add_spot(spot, True)
                    break # only adds to one section

    def add_negatives_to_sections(self) -> None:
        "used to find negative result spots to appropriate strip section"
        for type, section in zip(self.strip_sections.keys(), self.strip_sections.values()):
            if type == 'bkg':
                continue

            if len(section.spots) == 0:
                section.set_spots_manually()
            
            # set section's total rgb avg
            section.set_total_avg_rgb()

    def validate_results(self) -> None:
        "deals with test result potential positive, negative, false positive, error scenarios"
        
        results = self.get_section_results()
                
        #1 test is properly positive (bkg, test, and control line rgbs are > threshold)
        if False not in results[1:]:
            print("Test worked properly and result is positive")

        #2 test is properly negative (control line rgb is > threshold)
        elif results[1] == False & results[2] == True:
            print("Test worked properly and result is negative")

        #3 control error (bkg, and/or test line rgbs are > threshold)
        else:
            print("Test may not have worked properly")

    def get_section_results(self) -> list[bool]:
        "returs a list of booleans representing the result (positive or negative) of each section bkg, test, control"
        results = [] # bkg, test, control
        for strip in self.strip_sections.values():
            strip_result = False
            for spot in strip.spots:
                if spot["positive"]:
                    strip_result = True
                    break
            results.append(strip_result)
        return results

    def write_csv_row(self, filename:str = None) -> None:
        """
        writes the test results to csv file row in format:\n
        date, time, grid_index, block_type, bkg_r, bkg_g, bkg_b, test_r, test_g, test_b, cntrl_r, cntrl_g, cntrl_b"""

        # get current date and time
        now = datetime.now()

        # format date and time
        date = now.strftime("%m/%d/%Y")
        time = now.strftime("%H:%M:%S")

        # get rgb values of each section
        bkg_r, bkg_g, bkg_b = self.strip_sections['bkg'].total_avg_rgb
        test_r, test_g, test_b = self.strip_sections['test'].total_avg_rgb
        cntrl_r, cntrl_g, cntrl_b = self.strip_sections['control'].total_avg_rgb

        # create data to be written to csv
        data = [date, time, self.grid_index, bkg_r, bkg_g, bkg_b, test_r, test_g, 
            test_b, cntrl_r, cntrl_g, cntrl_b]

        # name of csv file
        if filename == None:
            filename = f"test_results_{date}_{time}.csv"

        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)

            # writing the data
            csvwriter.writerow(data)

class StripSection:
    "This class is responsible for holding data and processes regarding sections of test inner square (bkg, test, or control)"
    
    def init(self, test_square_img:np.ndarray, strip_type: str):
        bounds = self.get_bounds(test_square_img, strip_type)
        spots = [] # each spot is hashmap {"contour": np.ndarray, "avg_rgb": int, "positive": bool}
        total_avg_rgb = None

    def add_spot(self, spot:np.ndarray, b: bool) -> None:
        " adds spot to section as a hashmap with \"color\" and \"avg_rgb\" "

        index = len(self.spot)
        avg_rgb = get_rgb_avg_of_contour(spot)
        
        self.spot[index] = {
            "contour" : spot, 
            "avg_rgb" : avg_rgb,
            "positive" : b
        }

    def set_spots_manually(self):
        "mostly used to find negative result spots using ratios" 

        spot = ...
        self.add_spot(spot, False)
        ...        
    
    def set_total_avg_rgb(self, bkg = [0, 0, 0]) -> list[int]:
        "gets the total avg rgb by adding the spot rgb avgs together" 
        i = 0
        total_avg = [0, 0, 0]
        
        # adding the total avg with each spot avg
        for spot in self.spots.values():
            total_avg = list(map(lambda total, spot: total + spot, total_avg, spot["avg_rgb"]))
            i += 1
        
        # dividing by the number of spots
        total_avg = list(map(lambda total: total/i, total_avg))
        
        return total_avg

    # geometry 
    def get_bounds(self, test_square_img: np.ndarray, strip_type:str):

        if strip_type == "bkg":
            ...
        elif strip_type == "test":
            ...
        elif strip_type == "control":
            ...

    def bounds_contour(self, contour) -> bool:
        "checks if contour is within section bounds"
        ...
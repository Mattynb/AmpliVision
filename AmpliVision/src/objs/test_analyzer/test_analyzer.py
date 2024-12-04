import cv2 as cv
from datetime import datetime
from .strip_section import StripSection
from ..image.processors.image_processor import ColorContourExtractor

from matplotlib import pyplot as plt

class TestAnalyzer:
    "This class is responsible for getting and analyzing test results a.k.a phase B"

    def __init__(self, block):

        self.block = block

        # look only at the inner test square:
        self.test_square_img = block.get_test_area_img()

        # square used in csv export
        self.grid_index = block.index
        self.block_type = block.get_block_type()

        self.strip_sections = {
            "bkg": StripSection(self.test_square_img, 'bkg', block.rotation),
            "spot1": StripSection(self.test_square_img, 'spot1', block.rotation),
            "spot2": StripSection(self.test_square_img, 'spot2', block.rotation)
        }

    def analyze_test_result(self, double_thresh = False, display: bool = False):  # should I name it main?
        "gets test results from a block, analyses them, and export them to csv"

        # find the positive spots with hsv mask
        # need to think about cases where mask for example return one pixel.
        #   do you check for minimum contour size? do you only look for it manually? food for thought

        if display:
            print("rotation: ", self.block.rotation)

        # thresholds optimized for marker data
        rgb_spots = ColorContourExtractor.process_image(
            self.test_square_img, 
            hsv_lower=[0, 40, 20],
            double_thresh=double_thresh, 
            display=display
        )
        

        if display:
            cv.waitKey(100)
            plt.close()
            cv.destroyAllWindows()

        self.add_positives_to_sections(rgb_spots, display=display)

        # find the negative spots "manually" through ratios
        self.add_negatives_to_sections(display=display) 

        # get background color noise so we can remove it from other sections
        self.strip_sections['bkg'].set_total_avg_rgb()
        bkg_rgb_avg = self.strip_sections['bkg'].total_avg_rgb

        # remove background noise from other sections
        corrected_rgbs = []
        for section in self.strip_sections.values():
            if section.strip_type != 'bkg':

                if display:
                    print(f"{section.strip_type}AVG RGB: {section.total_avg_rgb}")
                    print("correcting: ", section.strip_type)

                corrected_rgbs.append(section.subtract_bkg(bkg_rgb_avg))

        if display:
            print("\n")

        # validate results to catch any potential errors in the test
        "TODO: adapt validate_results to work with the new strip configuration"
        # self.validate_results()

        # export results to csv
        row = self.create_csv_row(corrected_rgbs)
        return row
    

    def add_positives_to_sections(self, rgb_spots, display: int = 0) -> None:
        "used to add positive result spots to appropriate strip section"

        # adds each spot to its strip section
        for spot in rgb_spots:
            # display the spot
            cpy = cv.drawContours(self.test_square_img.copy(), [spot], -1, (0, 255, 0), 1)
            #plt.imshow(f'TA/add_positives_to_sections', cv.resize(cpy, (400, 400)))
            

            for section in self.strip_sections.values():
                if section.bounds_contour(spot):
                    section.add_spot(self.block, spot, True, debug=display)
                    # break # only adds to one section

    def add_negatives_to_sections(self, display: int = 0) -> None:
        "used to find negative result spots to appropriate strip section"
        for type, section in zip(self.strip_sections.keys(), self.strip_sections.values()):

            if len(section.spots) == 0:
                section.set_spots_manually(self.block, debug=display)

    def validate_results(self) -> None:
        "deals with test result potential positive, negative, false positive, error scenarios"

        results = self.get_section_results()

        # 1 test is properly positive (bkg, test, and spot2 line rgbs are > threshold)
        if results[1] & results[2]:
            print("Test worked properly and result is positive")

        # 2 test is properly negative (spot2 line rgb is > threshold)
        elif (not results[1]) & results[2]:
            print("Test worked properly and result is negative")

        # 3 spot2 error (bkg, and maybe test line rgbs are > threshold)
        else:
            print("Test may have not worked properly")

    def get_section_results(self) -> list[bool]:
        "returs a list of booleans representing the result (positive or negative) of each section bkg, test, spot2"
        results = []  # bkg, test, spot2
        for strip in self.strip_sections.values():
            strip_result = False

            # display
            # strip.print_spots()

            for spot in strip.spots:
                if spot["positive"] == True:
                    strip_result = True
                    break
            results.append(strip_result)

        print("\n")

        return results

    def create_csv_row(self, corrected_rgbs: list[list]) -> str:
        """
        writes the test results to csv file row in format:\n
        date, time, grid_index, block_type, bkg_r, bkg_g, bkg_b, test_r, test_g, test_b, cntrl_r, cntrl_g, cntrl_b"""

        # get current date and time
        now = datetime.now()

        # format date and time
        date = now.strftime("%m/%d/%Y")
        time = now.strftime("%H:%M:%S")

        # setting all rgb values to None
        bkg_r = " None"
        bkg_g, bkg_b = bkg_r, bkg_r
        spot1_r, spot1_g, spot1_b = bkg_r, bkg_r, bkg_r
        spot2_r, spot2_g, spot2_b = bkg_r, bkg_r, bkg_r

        # get rgb values of each section
        if self.strip_sections['bkg'].total_avg_rgb != None:
            bkg_b, bkg_g, bkg_r = self.strip_sections['bkg'].total_avg_rgb
            # print("bkg rgb: ", bkg_r, bkg_g, bkg_b)

        if self.strip_sections['spot1'].total_avg_rgb != None:
            spot1_b, spot1_g, spot1_r = self.strip_sections['spot1'].total_avg_rgb
            #print("spot1 rgb: ", test_r, test_g, test_b)

        if self.strip_sections["spot2"].total_avg_rgb != None:
            spot2_b, spot2_g, spot2_r = self.strip_sections['spot2'].total_avg_rgb

        spot1_corr_b, spot1_corr_g, spot1_corr_r = corrected_rgbs[0]
        spot2_corr_b, spot2_corr_g, spot2_corr_r = corrected_rgbs[1]

        # create data to be written to csv
        data = [
            date, time, self.grid_index, self.block_type,
            spot1_r, spot1_g, spot1_b,
            spot2_r, spot2_g, spot2_b,
            bkg_r, bkg_g, bkg_b,
            spot1_corr_r, spot1_corr_g, spot1_corr_b,
            spot2_corr_r, spot2_corr_g, spot2_corr_b
        ]

        return data

    def paint_spots(self, rgb_spot_results: dict[list]):
        """ colors the spots in the image accourding to the results 

        Args:
            rgb_spot_results (dict[list]): dictionary containing the rgb values of the spots in the image
            format: {r: [mean1, std1, mean2, std2], g: [mean1, std1, mean2, std2], b: [mean1, std1, mean2, std2]}
        """
        from numpy import random
        import time 

        # can show improved performance if get_rgb_avg_of_contour TODO is done
        t = time.time()
        self.analyze_test_result()
        #print(f"        analyze_test_result in {round(time.time() - t,2)}")

        image = self.test_square_img
        image_ = image.copy()
        for type, section in self.strip_sections.items():
            
            # dont paint bkg
            if section.strip_type == 'bkg':
                continue
            
            # get the correct index for result
            i = 0 if type == 'spot1' else 2

            rgb = []
            means = []
            for c in ('b', 'g', 'r'):
                mean, std = rgb_spot_results[c][i:i+2]
                means.append(mean)
                
                rgb.append(int(random.normal(mean, std)))
            rgb = tuple(rgb)
            
            if means == [0,0,0]:
                continue
            
            #print(f"Paiting {type} with {rgb}")
            image_ = section.paint_spot(image, rgb, display=False)
        
        self.block.set_test_area_img(image_)

        return self.block

        


"""TODO: Add a stripSection "middleman" class to take care of assigning which spots go to which section, etc"""

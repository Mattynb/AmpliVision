import cv2 as cv
from datetime import datetime
from .strip_section import StripSection
from ..image.processors.image_processor import ColorContourExtractor


class TestAnalyzer:
    "This class is responsible for getting and analyzing test results a.k.a phase B"

    def __init__(self, block):

        self.block = block
        
        # look only at the inner test square:
        self.test_square_img = block.get_test_area_img()

        # square used in csv export
        self.grid_index = block.index
        self.block_type = block.block_type

        self.strip_sections = {
            "bkg" : StripSection(self.test_square_img, 'bkg'), 
            "test" : StripSection(self.test_square_img, 'test'),
            "control" : StripSection(self.test_square_img, 'control')
        }

    def analyze_test_result(self): # should I name it main?
        "gets test results from a block, analyses them, and export them to csv"

        # find the positive spots with hsv mask
        # need to think about cases where mask for example return one pixel. 
        #   do you check for minimum contour size? do you only look for it manually? food for thought 
        rgb_spots = ColorContourExtractor.process_image(self.test_square_img) # hsv_lower= [...], hsv_upper= [...])
        
        copy = self.test_square_img.copy()
        cv.drawContours(copy, rgb_spots, -1, (0, 255, 0), 3)
        cv.imshow('analyze_test_result()', copy)
        cv.waitKey(0)
        cv.destroyAllWindows()

        print("spots found: ", len(rgb_spots))
        self.add_positives_to_sections(rgb_spots)

        # get background color noise so we can remove it from other sections
        #self.strip_sections['bkg'].set_total_avg_rgb()
        #bkg_rgb_avg = self.strip_sections['bkg'].total_avg_rgb

        # find the negative spots "manually" through ratios, removing bgk
        self.add_negatives_to_sections()

        # validate results to catch any potential errors in the test
        self.validate_results()

        for section in self.strip_sections.values():
            section.set_total_avg_rgb()
            print("total avg rgb in ", section.stip_type, " is: ", section.total_avg_rgb)
        
        # export results to csv
        return self.create_csv_row()

    def add_positives_to_sections(self, rgb_spots) -> None:
        "used to add positive result spots to appropriate strip section"

        # adds each spot to its strip section
        for spot in rgb_spots:
            for section in self.strip_sections.values():
                if section.bounds_contour(spot):
                    print("added spot to: ", section.stip_type)
                    section.add_spot(self.block, spot, True)
                    break # only adds to one section

    def add_negatives_to_sections(self) -> None:
        "used to find negative result spots to appropriate strip section"
        for type, section in zip(self.strip_sections.keys(), self.strip_sections.values()):

            if len(section.spots) == 0:
                section.set_spots_manually(self.block)

    def validate_results(self) -> None:
        "deals with test result potential positive, negative, false positive, error scenarios"
        
        results = self.get_section_results()
                
        #1 test is properly positive (bkg, test, and control line rgbs are > threshold)
        if results[1:]:
            print("Test worked properly and result is positive")

        #2 test is properly negative (control line rgb is > threshold)
        elif not results[1] & results[2]:
            print("Test worked properly and result is negative")

        #3 control error (bkg, and/or test line rgbs are > threshold)
        else:
            print("Test may have not worked properly")

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

    def create_csv_row(self) -> str:
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
        test_r, test_g, test_b = bkg_r, bkg_r, bkg_r
        cntrl_r, cntrl_g, cntrl_b = bkg_r, bkg_r, bkg_r

        # get rgb values of each section
        if self.strip_sections['bkg'].total_avg_rgb != None:
            bkg_r, bkg_g, bkg_b = self.strip_sections['bkg'].total_avg_rgb
            print("bkg rgb: ", bkg_r, bkg_g, bkg_b)

        if self.strip_sections['test'].total_avg_rgb != None:
            test_r, test_g, test_b = self.strip_sections['test'].total_avg_rgb
            print("test rgb: ", test_r, test_g, test_b)

        if self.strip_sections["control"].total_avg_rgb != None:
            cntrl_r, cntrl_g, cntrl_b = self.strip_sections['control'].total_avg_rgb
            print("control rgb: ", cntrl_r, cntrl_g, cntrl_b)

        # create data to be written to csv
        data = [date, time, self.grid_index, bkg_r, bkg_g, bkg_b, test_r, test_g, 
            test_b, cntrl_r, cntrl_g, cntrl_b]
        
        return data

"""TODO: Add a stripSection "middleman" class to take care of assigning which spots go to which section, etc"""
